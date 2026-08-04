"""M37 offline validator for side-lying foam-ring parameterized 3-D fits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def _draw_fit_overlay(
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
        eligible = bool(fit.get("eligible", False))
        selected = selected_instance_id == instance_id
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
        "M37.1: red box=crown grasp, white X=old rim-top, blue=visible arc ends",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


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
                "fit_ms": (fit.get("timing_ms") or {}).get("axis_template_fit_ms"),
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
) -> Dict[str, Any]:
    template_config = SideRingTemplateConfig.from_mapping(raw_config)
    matched_ids = _mouth_matches(instances, raw_config)
    requested = set(int(value) for value in (instance_ids or []))
    fits: List[Dict[str, Any]] = []
    for instance in instances:
        if instance.class_name != "foam_ring":
            continue
        if requested and int(instance.instance_id) not in requested:
            continue
        mouth_matched = int(instance.instance_id) in matched_ids
        if mouth_matched and not include_mouth_matched:
            continue
        fits.append(
            fit_side_ring_instance(
                instance,
                depth_mm,
                intrinsics,
                template_config,
                mouth_matched=mouth_matched,
            )
        )

    selected = select_best_side_ring(fits)
    selected_id = int(selected["ring_instance_id"]) if selected is not None else None
    output_dir = output_root / capture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = _draw_fit_overlay(rgb_bgr, instances, fits, intrinsics, selected_id)
    cv2.imwrite(str(output_dir / "side_ring_template_overlay.jpg"), overlay)

    if save_ply:
        for fit in fits:
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

    payload = {
        "schema_version": "1.1",
        "message_type": "side_ring_template_offline_validation_result",
        "stage": "M37.1_near_visible_cylindrical_crown_grasp_point_correction",
        "status": "ok",
        "capture_id": capture_id,
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
        "evaluated_count": len(fits),
        "eligible_count": sum(1 for item in fits if item.get("eligible")),
        "selected_ring_instance_id": selected_id,
        "selected": _strip_debug(selected) if selected is not None else None,
        "fits": [_strip_debug(item) for item in fits],
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
        "stage": "M37.1",
        "status": "ok",
        "captures": len(payloads),
        "evaluated_count": sum(int(item.get("evaluated_count", 0)) for item in payloads),
        "eligible_count": sum(int(item.get("eligible_count", 0)) for item in payloads),
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
    print("[PASS] M37.1 near-side visible crown grasp-point correction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
