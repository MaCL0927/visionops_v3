"""M37 offline validator for side-lying foam-ring parameterized 3-D fits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    associate_ring_mouths,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import (  # noqa: E402
    CapturePaths,
    discover_captures,
    load_rgb_depth_meta,
    load_yaml,
    resolve_capture_paths,
    resolve_intrinsics,
    write_ascii_ply,
    write_json,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (  # noqa: E402
    SegmentationInstance,
    infer_ultralytics_segmentation,
    load_yolo_segmentation_labels,
    runtime_result_to_segmentation_instances,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_template import (  # noqa: E402
    SideRingTemplateConfig,
    fit_side_ring_instance,
    select_best_side_ring,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "line.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "foam_ring_side_template_m37"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M37：侧躺泡沫圆环参数化短圆柱3D模板拟合离线验证",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="M36.5 save_debug目录，含exact_rgb/depth和Runtime/Geometry JSON",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--capture-id")
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--class-names", default="foam_ring,ring_mouth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-mouth-matched", action="store_true")
    parser.add_argument("--instance-id", type=int, action="append")
    parser.add_argument(
        "--mode",
        choices=("first_valid_confidence", "exhaustive"),
        default=None,
        help="默认读取side_ring_template.execution_mode",
    )
    parser.add_argument(
        "--search-profile",
        choices=("auto", "fast", "accurate"),
        default=None,
        help="first-valid默认auto，exhaustive默认accurate",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-ply", action="store_true")
    return parser


def _strip_debug(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_debug(item)
            for key, item in value.items()
            if str(key) != "_debug"
        }
    if isinstance(value, list):
        return [_strip_debug(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_debug(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _project_polyline(
    points: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = points[:, 2] > 1e-6
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = (
        float(intrinsics["fx"]) * points[valid, 0] / points[valid, 2]
        + float(intrinsics["cx"])
    )
    pixels[valid, 1] = (
        float(intrinsics["fy"]) * points[valid, 1] / points[valid, 2]
        + float(intrinsics["cy"])
    )
    return pixels


def draw_side_ring_fit_overlay(
    rgb_bgr: np.ndarray,
    instances: Sequence[SegmentationInstance],
    fits: Sequence[Mapping[str, Any]],
    intrinsics: Mapping[str, float],
    selected_instance_id: int | None,
) -> np.ndarray:
    overlay = rgb_bgr.copy()
    instance_map = {int(item.instance_id): item for item in instances}
    for fit in fits:
        instance_id = int(fit.get("ring_instance_id", -1))
        instance = instance_map.get(instance_id)
        if instance is None:
            continue
        processing_status = str(fit.get("processing_status") or "evaluated")
        eligible_value = fit.get("eligible")
        eligible = bool(eligible_value) if eligible_value is not None else False
        selected = selected_instance_id == instance_id
        if processing_status == "deferred":
            color = (128, 128, 128)
        else:
            color = (0, 255, 0) if eligible else (0, 165, 255)
        if selected:
            color = (255, 255, 0)
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
        x1, y1, _, _ = instance.bbox_xyxy
        if processing_status == "deferred":
            label = "S%d D %.2f" % (
                instance_id, float(fit.get("ring_confidence", 0.0))
            )
        else:
            label = "S%d %.2f" % (instance_id, float(fit.get("fit_score", 0.0)))
        cv2.putText(
            overlay,
            label,
            (int(x1), max(12, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        if fit.get("center_uv") is None:
            continue
        center = tuple(int(round(float(value))) for value in fit["center_uv"])
        near = tuple(int(round(float(value))) for value in fit["near_opening_center_uv"])
        far = tuple(int(round(float(value))) for value in fit["far_opening_center_uv"])
        crown = fit.get("near_side_crown") or fit.get("top_arc") or {}
        grasp = tuple(
            int(round(float(value)))
            for value in crown.get("grasp_point_uv", [0, 0])
        )
        legacy_rim = fit.get("near_opening_rim_top_diagnostic") or {}
        legacy_uv_raw = legacy_rim.get("point_uv")
        cv2.line(overlay, far, near, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(overlay, center, near, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(overlay, near, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, far, 4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(
            overlay,
            (grasp[0] - 5, grasp[1] - 5),
            (grasp[0] + 5, grasp[1] + 5),
            (0, 0, 255),
            2,
        )
        if isinstance(legacy_uv_raw, (list, tuple)) and len(legacy_uv_raw) >= 2:
            legacy_uv = tuple(int(round(float(value))) for value in legacy_uv_raw[:2])
            cv2.drawMarker(
                overlay,
                legacy_uv,
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=8,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
        visible_arc = crown.get("visible_arc") or {}
        for key in ("upper_endpoint_uv", "lower_endpoint_uv"):
            endpoint = visible_arc.get(key)
            if isinstance(endpoint, (list, tuple)) and len(endpoint) >= 2:
                point = tuple(int(round(float(value))) for value in endpoint[:2])
                cv2.circle(overlay, point, 3, (255, 0, 0), -1, cv2.LINE_AA)
        debug = fit.get("_debug") or {}
        for key, circle_color in (
            ("near_outer_circle_camera_mm", (0, 255, 255)),
            ("near_inner_circle_camera_mm", (255, 255, 0)),
            ("grasp_outer_circle_camera_mm", (0, 165, 255)),
        ):
            points = debug.get(key)
            if points is None:
                continue
            pixels = _project_polyline(np.asarray(points), intrinsics)
            finite = np.isfinite(pixels).all(axis=1)
            if int(np.count_nonzero(finite)) >= 3:
                contour = np.rint(pixels[finite]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [contour], True, circle_color, 1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "M37.2: confidence-first, first valid exits; gray=deferred",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


# Backward-compatible private alias used by early M37 scripts/tests.
_draw_fit_overlay = draw_side_ring_fit_overlay

def _mouth_matches(
    instances: Sequence[SegmentationInstance],
    raw_config: Mapping[str, Any],
) -> set[int]:
    rings = [item for item in instances if item.class_name == "foam_ring"]
    mouths = [item for item in instances if item.class_name == "ring_mouth"]
    matches, _ = associate_ring_mouths(rings, mouths, GeometryConfig(dict(raw_config)))
    return {int(ring.instance_id) for ring, _, _ in matches}


def _summary_rows(capture_id: str, fits: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for fit in fits:
        top_arc = fit.get("near_side_crown") or fit.get("top_arc") or {}
        axis = fit.get("axis_toward_camera") or [None, None, None]
        point = top_arc.get("grasp_point_camera_mm") or [None, None, None]
        uv = top_arc.get("grasp_point_uv") or [None, None]
        rows.append(
            {
                "capture_id": capture_id,
                "ring_instance_id": fit.get("ring_instance_id"),
                "ring_confidence": fit.get("ring_confidence"),
                "attempt_rank": fit.get("attempt_rank"),
                "processing_status": fit.get("processing_status"),
                "eligible": fit.get("eligible"),
                "mouth_matched": fit.get("mouth_matched"),
                "rejection_reasons": ";".join(fit.get("rejection_reasons") or []),
                "fit_score": fit.get("fit_score"),
                "radial_inlier_ratio": fit.get("radial_inlier_ratio"),
                "radial_residual_median_mm": fit.get("radial_residual_median_mm"),
                "radial_residual_p90_mm": fit.get("radial_residual_p90_mm"),
                "observed_axis_span_mm": fit.get("observed_axis_span_mm"),
                "axis_view_angle_deg": fit.get("axis_view_angle_deg"),
                "near_distance_mm": fit.get("near_endpoint_camera_distance_mm"),
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "grasp_x_mm": point[0],
                "grasp_y_mm": point[1],
                "grasp_z_mm": point[2],
                "grasp_u": uv[0],
                "grasp_v": uv[1],
                "grasp_direction_source": top_arc.get("direction_source"),
                "visible_arc_span_deg": (top_arc.get("visible_arc") or {}).get(
                    "visible_angular_span_deg"
                ),
                "search_profile_used": fit.get("search_profile_used"),
                "accurate_fallback_used": fit.get("accurate_fallback_used"),
                "fit_ms": (fit.get("timing_ms") or {}).get("axis_template_fit_ms"),
                "total_ms": (fit.get("timing_ms") or {}).get("total_ms"),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



def _ordered_side_ring_candidates(
    instances: Sequence[SegmentationInstance],
    *,
    matched_ids: set[int],
    include_mouth_matched: bool,
    requested_ids: set[int],
) -> List[Tuple[SegmentationInstance, bool]]:
    candidates: List[Tuple[SegmentationInstance, bool]] = []
    for instance in instances:
        if instance.class_name != "foam_ring":
            continue
        instance_id = int(instance.instance_id)
        if requested_ids and instance_id not in requested_ids:
            continue
        mouth_matched = instance_id in matched_ids
        if mouth_matched and not include_mouth_matched:
            continue
        candidates.append((instance, mouth_matched))
    candidates.sort(
        key=lambda item: (
            -float(item[0].confidence),
            int(item[0].instance_id),
        )
    )
    return candidates


def _deferred_fit_record(
    instance: SegmentationInstance,
    *,
    mouth_matched: bool,
    attempt_rank: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "ring_instance_id": int(instance.instance_id),
        "ring_confidence": float(instance.confidence),
        "ring_bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "mouth_matched": bool(mouth_matched),
        "attempt_rank": int(attempt_rank),
        "processing_status": "deferred",
        "deferred_reason": str(reason),
        "eligible": None,
        "rejection_reasons": [],
        "timing_ms": {"total_ms": 0.0},
    }

def _process(
    *,
    capture_id: str,
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    instances: Sequence[SegmentationInstance],
    raw_config: Dict[str, Any],
    output_root: Path,
    include_mouth_matched: bool,
    instance_ids: Sequence[int] | None,
    save_ply: bool,
    inputs: Mapping[str, Any],
    execution_mode: str | None = None,
    search_profile: str | None = None,
) -> Dict[str, Any]:
    process_started = time.perf_counter()
    template_config = SideRingTemplateConfig.from_mapping(raw_config)
    mode = str(execution_mode or template_config.execution_mode).strip().lower()
    if mode not in {"first_valid_confidence", "exhaustive"}:
        raise ValueError("M37 execution mode must be first_valid_confidence or exhaustive")
    effective_search_profile = str(
        search_profile
        or (
            template_config.first_valid_search_profile
            if mode == "first_valid_confidence"
            else template_config.exhaustive_search_profile
        )
    ).strip().lower()
    if effective_search_profile not in {"auto", "fast", "accurate"}:
        raise ValueError("M37 search profile must be auto, fast or accurate")

    association_started = time.perf_counter()
    matched_ids = _mouth_matches(instances, raw_config)
    association_ms = (time.perf_counter() - association_started) * 1000.0
    requested = set(int(value) for value in (instance_ids or []))
    order_started = time.perf_counter()
    candidates = _ordered_side_ring_candidates(
        instances,
        matched_ids=matched_ids,
        include_mouth_matched=include_mouth_matched,
        requested_ids=requested,
    )
    candidate_order_ms = (time.perf_counter() - order_started) * 1000.0

    fits: List[Dict[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    fit_loop_started = time.perf_counter()
    processed_count = 0
    stop_index: int | None = None
    for index, (instance, mouth_matched) in enumerate(candidates):
        attempt_rank = index + 1
        maximum_attempts = int(template_config.maximum_instances_to_attempt)
        if maximum_attempts > 0 and processed_count >= maximum_attempts:
            stop_index = index
            break
        fit = fit_side_ring_instance(
            instance,
            depth_mm,
            intrinsics,
            template_config,
            mouth_matched=mouth_matched,
            search_profile=effective_search_profile,
        )
        fit["attempt_rank"] = int(attempt_rank)
        fit["processing_status"] = "evaluated"
        fits.append(fit)
        processed_count += 1
        if (
            mode == "first_valid_confidence"
            and template_config.stop_after_first_eligible
            and bool(fit.get("eligible", False))
        ):
            selected = fit
            stop_index = index + 1
            break

    fit_loop_ms = (time.perf_counter() - fit_loop_started) * 1000.0
    if mode == "exhaustive":
        selected = select_best_side_ring(fits)
    elif selected is None:
        selected = select_best_side_ring(fits)

    if stop_index is not None and stop_index < len(candidates):
        reason = (
            "after_first_valid_confidence_candidate"
            if selected is not None and bool(selected.get("eligible", False))
            else "maximum_instances_to_attempt_reached"
        )
        for index in range(stop_index, len(candidates)):
            instance, mouth_matched = candidates[index]
            fits.append(
                _deferred_fit_record(
                    instance,
                    mouth_matched=mouth_matched,
                    attempt_rank=index + 1,
                    reason=reason,
                )
            )

    selected_id = int(selected["ring_instance_id"]) if selected is not None else None
    output_started = time.perf_counter()
    output_dir = output_root / capture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_side_ring_fit_overlay(rgb_bgr, instances, fits, intrinsics, selected_id)
    cv2.imwrite(str(output_dir / "side_ring_template_overlay.jpg"), overlay)

    if save_ply:
        ply_fits = (
            [selected]
            if mode == "first_valid_confidence" and selected is not None
            else [fit for fit in fits if fit.get("processing_status") == "evaluated"]
        )
        for fit in ply_fits:
            if fit is None:
                continue
            debug = fit.get("_debug") or {}
            points = debug.get("trimmed_points_camera_mm")
            if points is not None:
                write_ascii_ply(
                    output_dir / ("ring_%02d_scene_points.ply" % int(fit["ring_instance_id"])),
                    np.asarray(points),
                )
            template_parts = []
            for key in (
                "near_outer_circle_camera_mm",
                "near_inner_circle_camera_mm",
                "far_outer_circle_camera_mm",
                "grasp_outer_circle_camera_mm",
            ):
                if debug.get(key) is not None:
                    template_parts.append(np.asarray(debug[key]))
            if template_parts:
                write_ascii_ply(
                    output_dir / ("ring_%02d_template_curves.ply" % int(fit["ring_instance_id"])),
                    np.vstack(template_parts),
                )

    output_generation_ms = (time.perf_counter() - output_started) * 1000.0
    evaluated_fits = [
        fit for fit in fits if fit.get("processing_status") == "evaluated"
    ]
    deferred_fits = [
        fit for fit in fits if fit.get("processing_status") == "deferred"
    ]
    process_total_ms = (time.perf_counter() - process_started) * 1000.0
    payload = {
        "schema_version": "1.2",
        "message_type": "side_ring_template_offline_validation_result",
        "stage": "M37.2_confidence_first_fast_side_ring_template",
        "status": "ok",
        "capture_id": capture_id,
        "execution": {
            "mode": mode,
            "candidate_order_rule": "foam_ring_confidence_descending",
            "search_profile": effective_search_profile,
            "stop_after_first_eligible": bool(template_config.stop_after_first_eligible),
            "maximum_instances_to_attempt": int(template_config.maximum_instances_to_attempt),
            "first_valid_early_exit_triggered": bool(
                mode == "first_valid_confidence"
                and selected is not None
                and len(deferred_fits) > 0
            ),
        },
        "inputs": dict(inputs),
        "image": {"width": int(rgb_bgr.shape[1]), "height": int(rgb_bgr.shape[0])},
        "depth": {"dtype": str(depth_mm.dtype)},
        "intrinsics": {key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")},
        "template": {
            "outer_radius_mm": template_config.outer_radius_mm,
            "inner_radius_mm": template_config.inner_radius_mm,
            "axial_length_mm": template_config.axial_length_mm,
            "axis_sign_rule": "choose endpoint with smaller camera-origin distance",
            "grasp_point_rule": (
                "outer cylindrical visible-arc point at configured near-opening "
                "axial inset; old projected rim-top retained for diagnostics"
            ),
            "grasp_radius_mode": template_config.grasp_radius_mode,
            "grasp_axial_inset_mm": template_config.grasp_axial_inset_mm,
            "visible_crown_upper_fraction": template_config.visible_crown_upper_fraction,
        },
        "rings_detected": sum(1 for item in instances if item.class_name == "foam_ring"),
        "mouths_detected": sum(1 for item in instances if item.class_name == "ring_mouth"),
        "mouth_matched_ring_ids": sorted(matched_ids),
        "candidate_count": len(candidates),
        "candidate_order": [
            {
                "rank": index + 1,
                "ring_instance_id": int(instance.instance_id),
                "confidence": float(instance.confidence),
            }
            for index, (instance, _) in enumerate(candidates)
        ],
        "evaluated_count": len(evaluated_fits),
        "deferred_count": len(deferred_fits),
        "eligible_count": sum(1 for item in evaluated_fits if item.get("eligible")),
        "attempted_instance_ids": [
            int(item["ring_instance_id"]) for item in evaluated_fits
        ],
        "selected_ring_instance_id": selected_id,
        "selected": _strip_debug(selected) if selected is not None else None,
        "fits": [_strip_debug(item) for item in fits],
        "timing_ms": {
            "association_ms": float(association_ms),
            "candidate_filter_sort_ms": float(candidate_order_ms),
            "fit_loop_ms": float(fit_loop_ms),
            "output_generation_ms": float(output_generation_ms),
            "total_ms": float(process_total_ms),
            "evaluated_instance_total_ms": float(
                sum(
                    float((item.get("timing_ms") or {}).get("total_ms", 0.0))
                    for item in evaluated_fits
                )
            ),
        },
        "files": {
            "overlay": str(output_dir / "side_ring_template_overlay.jpg"),
            "result": str(output_dir / "side_ring_template_result.json"),
        },
    }
    write_json(output_dir / "side_ring_template_result.json", payload)
    _write_csv(output_dir / "side_ring_template_summary.csv", _summary_rows(capture_id, fits))
    return payload


def _bundle_input(
    bundle: Path,
) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, float], List[SegmentationInstance], Dict[str, Any]]:
    bundle = bundle.expanduser().resolve()
    rgb_path = bundle / "exact_rgb.png"
    depth_path = bundle / "exact_depth.png"
    runtime_path = bundle / "runtime_inference_result.json"
    geometry_path = bundle / "online_geometry_result.json"
    for path in (rgb_path, depth_path, runtime_path, geometry_path):
        if not path.exists():
            raise FileNotFoundError(f"M36 bundle缺少文件: {path}")
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise ValueError("无法读取bundle RGB/Depth")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    coordinate = geometry.get("coordinate_space") or {}
    roi = tuple(int(value) for value in coordinate.get("geometry_roi_xyxy", []))
    full_size = coordinate.get("full_frame_size") or {}
    if len(roi) != 4:
        raise ValueError("online_geometry_result缺少geometry_roi_xyxy")
    adaptation = runtime_result_to_segmentation_instances(
        runtime,
        (int(full_size.get("height", 720)), int(full_size.get("width", 1280))),
        require_proto_mask=True,
        reject_bbox_fallback=True,
        minimum_mask_area_px=20,
        geometry_roi_xyxy=roi,
    )
    intrinsics = {key: float(geometry["intrinsics"][key]) for key in ("fx", "fy", "cx", "cy")}
    capture_id = str(geometry.get("capture_timestamp_ms") or bundle.name)
    inputs = {
        "mode": "m36_debug_bundle",
        "bundle": str(bundle),
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "runtime_result": str(runtime_path),
        "geometry_result": str(geometry_path),
    }
    return capture_id, rgb, depth, intrinsics, adaptation.instances, inputs


def _raw_instances(
    rgb: np.ndarray,
    paths: CapturePaths,
    args: argparse.Namespace,
    raw_config: Mapping[str, Any],
) -> List[SegmentationInstance]:
    names = [value.strip() for value in str(args.class_names).split(",") if value.strip()]
    labels = args.labels
    if labels is None and args.labels_dir is not None:
        labels = args.labels_dir / (paths.capture_id + ".txt")
    if labels is not None:
        return load_yolo_segmentation_labels(labels, rgb.shape[:2], names)
    if args.model is None:
        raise ValueError("raw数据模式必须提供--model或--labels/--labels-dir")
    inference = raw_config.get("inference") or {}
    device = args.device if args.device is not None else str(inference.get("device") or "auto")
    return infer_ultralytics_segmentation(
        rgb,
        args.model,
        confidence=float(inference.get("confidence", 0.20)),
        iou=float(inference.get("iou", 0.50)),
        image_size=int(inference.get("image_size", 640)),
        max_detections=int(inference.get("max_detections", 300)),
        retina_masks=bool(inference.get("retina_masks", True)),
        device=device,
    )


def _raw_capture_paths(args: argparse.Namespace) -> List[CapturePaths]:
    if args.rgb or args.depth or args.meta:
        if not (args.rgb and args.depth and args.meta):
            raise ValueError("单帧raw模式必须同时提供--rgb、--depth、--meta")
        capture_id = args.capture_id or args.rgb.stem
        return [CapturePaths(capture_id, args.rgb, args.depth, args.meta)]
    if args.data_root is None:
        raise ValueError("必须提供--bundle，或提供--data-root/--rgb+--depth+--meta")
    if args.capture_id:
        return [resolve_capture_paths(args.data_root, args.capture_id)]
    captures = discover_captures(args.data_root)
    if not args.all:
        captures = captures[:1]
    if args.limit > 0:
        captures = captures[: args.limit]
    return captures


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    raw_config = load_yaml(config_path)
    output_root = args.output.expanduser().resolve()
    payloads = []

    if args.bundle is not None:
        capture_id, rgb, depth, intrinsics, instances, inputs = _bundle_input(args.bundle)
        payloads.append(
            _process(
                capture_id=capture_id,
                rgb_bgr=rgb,
                depth_mm=depth,
                intrinsics=intrinsics,
                instances=instances,
                raw_config=raw_config,
                output_root=output_root,
                include_mouth_matched=bool(args.include_mouth_matched),
                instance_ids=args.instance_id,
                save_ply=not args.no_ply,
                inputs=inputs,
                execution_mode=args.mode,
                search_profile=args.search_profile,
            )
        )
    else:
        for paths in _raw_capture_paths(args):
            rgb, depth, meta = load_rgb_depth_meta(paths)
            intrinsics = resolve_intrinsics(meta, rgb.shape[1], rgb.shape[0])
            instances = _raw_instances(rgb, paths, args, raw_config)
            payloads.append(
                _process(
                    capture_id=paths.capture_id,
                    rgb_bgr=rgb,
                    depth_mm=depth,
                    intrinsics=intrinsics,
                    instances=instances,
                    raw_config=raw_config,
                    output_root=output_root,
                    include_mouth_matched=bool(args.include_mouth_matched),
                    instance_ids=args.instance_id,
                    save_ply=not args.no_ply,
                    execution_mode=args.mode,
                    search_profile=args.search_profile,
                    inputs={
                        "mode": "raw_capture",
                        "rgb": str(paths.rgb),
                        "depth": str(paths.depth),
                        "meta": str(paths.meta),
                        "model": str(args.model) if args.model else None,
                    },
                )
            )

    summary = {
        "stage": "M37.2",
        "status": "ok",
        "captures": len(payloads),
        "evaluated_count": sum(int(item.get("evaluated_count", 0)) for item in payloads),
        "eligible_count": sum(int(item.get("eligible_count", 0)) for item in payloads),
        "deferred_count": sum(int(item.get("deferred_count", 0)) for item in payloads),
        "total_processing_ms": sum(
            float((item.get("timing_ms") or {}).get("total_ms", 0.0))
            for item in payloads
        ),
        "selected": [
            {
                "capture_id": item.get("capture_id"),
                "ring_instance_id": item.get("selected_ring_instance_id"),
                "grasp_point_uv": (
                    ((item.get("selected") or {}).get("near_side_crown") or {}).get(
                        "grasp_point_uv"
                    )
                    or ((item.get("selected") or {}).get("top_arc") or {}).get(
                        "grasp_point_uv"
                    )
                ),
            }
            for item in payloads
        ],
        "output": str(output_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[PASS] M37.2 confidence-first fast side-ring fitting completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
