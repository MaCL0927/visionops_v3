#!/usr/bin/env python3
"""Replay saved RGB-D bundles through M38.3 branch B only.

The diagnostic evaluates two evidence sources:

* a segmented mouth that is genuinely off-centre inside its ring;
* a depth-inferred deep aperture inside an unmatched ring.

Eligible evidence is solved with the projection-constrained fixed-radius local
cylinder and then passed through the retained rim-pinch/collision evaluator.
M38.1, M36 and M37.6 are intentionally skipped so branch-B behavior can be
inspected independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    _associate_ring_mouths_detailed,
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (  # noqa: E402
    _replace_mouth_instances,
    _replace_ring_instances,
    _scoped_m38b_geometry_config,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import (  # noqa: E402
    load_yaml,
    write_json,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (  # noqa: E402
    _strip_debug,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.partial_opening_cylinder_m383 import (  # noqa: E402
    fit_partial_opening_cylinder_m383,
    infer_depth_partial_opening,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (  # noqa: E402
    SegmentationInstance,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import (  # noqa: E402
    _bundle_input,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.visualization import (  # noqa: E402
    draw_overlay,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing saved capture bundles")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("production/foam_ring_grasp/config/line.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/m38_3_branch_b_replay"))
    parser.add_argument("--capture", action="append", default=[])
    return parser.parse_args()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _row(capture_id: str, fit: Mapping[str, Any], source: str) -> Dict[str, Any]:
    diagnostics = fit.get("diagnostics") if isinstance(fit.get("diagnostics"), Mapping) else {}
    timing = fit.get("timing_ms") if isinstance(fit.get("timing_ms"), Mapping) else {}
    return {
        "capture_id": capture_id,
        "ring_instance_id": fit.get("ring_instance_id"),
        "mouth_instance_id": fit.get("mouth_instance_id"),
        "opening_evidence_source": source,
        "eligible_local_fit": bool(fit.get("eligible", False)),
        "opening_depth_gap_mm": diagnostics.get("opening_depth_gap_mm"),
        "opening_area_ratio": diagnostics.get("opening_area_ratio"),
        "opening_center_offset_ratio": diagnostics.get("opening_center_offset_ratio"),
        "opening_boundary_contact_ratio": diagnostics.get("opening_boundary_contact_ratio"),
        "radial_inlier_ratio": diagnostics.get("radial_inlier_ratio"),
        "radial_residual_median_mm": diagnostics.get("radial_residual_median_mm"),
        "radial_residual_p90_mm": diagnostics.get("radial_residual_p90_mm"),
        "normal_axis_median_deg": diagnostics.get("normal_axis_median_deg"),
        "normal_axis_p90_deg": diagnostics.get("normal_axis_p90_deg"),
        "projected_axis_error_deg": diagnostics.get("projected_axis_error_deg"),
        "observed_axis_span_mm": diagnostics.get("observed_axis_span_mm"),
        "axis_view_angle_deg": diagnostics.get("axis_view_angle_deg"),
        "rim_support_point_count": diagnostics.get("rim_support_point_count"),
        "rim_anchor_residual_p90_mm": diagnostics.get("rim_anchor_residual_p90_mm"),
        "opening_center_error_px": diagnostics.get("opening_center_error_px"),
        "synthetic_opening_coverage": diagnostics.get("synthetic_opening_coverage"),
        "candidate_axis_count": diagnostics.get("candidate_axis_count"),
        "fit_total_ms": timing.get("total_ms"),
        "rejection_reasons": ";".join(str(value) for value in fit.get("rejection_reasons") or []),
    }


def _paint_fit_masks(image: np.ndarray, fits: List[Mapping[str, Any]]) -> np.ndarray:
    output = image.copy()
    for fit in fits:
        debug = fit.get("_debug") if isinstance(fit.get("_debug"), Mapping) else {}
        deep = debug.get("deep_candidate_mask")
        opening = debug.get("opening_mask")
        side = debug.get("side_surface_mask")
        rim = debug.get("endpoint_support_mask")
        synthetic = debug.get("synthetic_mouth_mask")
        synthetic_outer = debug.get("synthetic_outer_mask")
        for mask, color, alpha in (
            (deep, (0, 128, 255), 0.18),
            (side, (255, 0, 255), 0.22),
            (opening, (255, 0, 0), 0.25),
        ):
            if isinstance(mask, np.ndarray) and mask.shape == output.shape[:2] and np.any(mask):
                layer = output.copy()
                layer[mask.astype(bool)] = color
                output = cv2.addWeighted(layer, alpha, output, 1.0 - alpha, 0.0)
        if isinstance(rim, np.ndarray) and rim.shape == output.shape[:2] and np.any(rim):
            contours, _ = cv2.findContours(rim.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
        if isinstance(synthetic_outer, np.ndarray) and synthetic_outer.shape == output.shape[:2] and np.any(synthetic_outer):
            contours, _ = cv2.findContours(synthetic_outer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        if isinstance(synthetic, np.ndarray) and synthetic.shape == output.shape[:2] and np.any(synthetic):
            contours, _ = cv2.findContours(synthetic.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    return output


def main() -> int:
    args = _parse_args()
    raw = load_yaml(args.config.resolve())
    geometry_config = GeometryConfig(dict(raw))
    branch_cfg = raw.get("m38_branch_b") if isinstance(raw.get("m38_branch_b"), Mapping) else {}
    minimum_offset_ratio = _safe_float(branch_cfg.get("minimum_partial_opening_center_offset_ratio"), 0.10)
    maximum_candidates = max(1, int(branch_cfg.get("maximum_candidates_per_trigger", 4)))
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_captures = {str(value) for value in args.capture}

    rows: List[Dict[str, Any]] = []
    captures: List[Dict[str, Any]] = []
    for bundle in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected_captures and bundle.name not in selected_captures:
            continue
        capture_id, rgb, depth, intrinsics, instances, _ = _bundle_input(bundle)
        rings = [item for item in instances if item.class_name == "foam_ring"]
        mouths = [item for item in instances if item.class_name == "ring_mouth"]
        matches, unmatched_rings, _, association_debug = _associate_ring_mouths_detailed(
            rings, mouths, geometry_config
        )

        evidence_rows: List[Dict[str, Any]] = []
        for ring, mouth, metrics in matches:
            x1, y1, x2, y2 = ring.bbox_xyxy
            diagonal = max(1.0, math.hypot(float(x2 - x1), float(y2 - y1)))
            offset_ratio = float(
                np.linalg.norm(np.asarray(mouth.centroid_uv) - np.asarray(ring.centroid_uv))
            ) / diagonal
            if offset_ratio < minimum_offset_ratio:
                continue
            evidence_rows.append({
                "ring": ring,
                "mouth": mouth,
                "association": dict(metrics),
                "inferred": None,
                "source": "segmented_partial_mouth",
                "score": _safe_float(metrics.get("association_score"), 0.0),
            })
        inference_records: List[Dict[str, Any]] = []
        inference_started = time.perf_counter()
        for ring in unmatched_rings:
            evidence = infer_depth_partial_opening(ring, depth, raw)
            inference_records.append({
                "ring_instance_id": int(ring.instance_id),
                "eligible": bool(evidence.get("eligible", False)),
                "rejection_reasons": list(evidence.get("rejection_reasons") or []),
                "diagnostics": dict(evidence.get("diagnostics") or {}),
                "timing_ms": dict(evidence.get("timing_ms") or {}),
            })
            mouth = evidence.get("mouth_instance")
            if bool(evidence.get("eligible", False)) and isinstance(mouth, SegmentationInstance):
                diagnostics = evidence.get("diagnostics") if isinstance(evidence.get("diagnostics"), Mapping) else {}
                evidence_rows.append({
                    "ring": ring,
                    "mouth": mouth,
                    "association": dict(evidence.get("association") or {}),
                    "inferred": evidence,
                    "source": "depth_inferred_partial_opening",
                    "score": _safe_float(diagnostics.get("evidence_score"), 0.0),
                })
        inference_wall_ms = (time.perf_counter() - inference_started) * 1000.0
        evidence_rows.sort(key=lambda item: (-float(item["score"]), -float(item["ring"].confidence), int(item["ring"].instance_id)))
        evidence_rows = evidence_rows[:maximum_candidates]

        fits: List[Dict[str, Any]] = []
        fit_sources: Dict[int, str] = {}
        pose_by_ring: Dict[int, Mapping[str, Any]] = {}
        mouth_replacements: Dict[int, SegmentationInstance] = {}
        ring_replacements: Dict[int, SegmentationInstance] = {}
        fit_started = time.perf_counter()
        for evidence in evidence_rows:
            ring = evidence["ring"]
            mouth = evidence["mouth"]
            fit = fit_partial_opening_cylinder_m383(
                ring,
                mouth,
                evidence["association"],
                rings,
                depth,
                intrinsics,
                raw,
                inferred_opening=evidence["inferred"],
            )
            fits.append(fit)
            source = str(evidence["source"])
            fit_sources[int(ring.instance_id)] = source
            rows.append(_row(capture_id, fit, source))
            payload = fit.get("pose_payload")
            synthetic_mouth = fit.get("synthetic_mouth_instance")
            synthetic_ring = fit.get("synthetic_ring_instance")
            if (
                bool(fit.get("eligible"))
                and isinstance(payload, Mapping)
                and isinstance(synthetic_mouth, SegmentationInstance)
                and isinstance(synthetic_ring, SegmentationInstance)
            ):
                pose_by_ring[int(ring.instance_id)] = dict(payload)
                mouth_replacements[int(synthetic_mouth.instance_id)] = synthetic_mouth
                ring_replacements[int(synthetic_ring.instance_id)] = synthetic_ring
        fit_wall_ms = (time.perf_counter() - fit_started) * 1000.0

        geometry_wall_ms = 0.0
        if pose_by_ring:
            branch_instances = _replace_ring_instances(instances, ring_replacements)
            branch_instances = _replace_mouth_instances(branch_instances, mouth_replacements)
            geometry_started = time.perf_counter()
            scene = analyze_scene(
                branch_instances,
                depth,
                intrinsics,
                _scoped_m38b_geometry_config(
                    geometry_config, sorted(pose_by_ring), pose_by_ring
                ),
            )
            geometry_wall_ms = (time.perf_counter() - geometry_started) * 1000.0
        else:
            scene = {
                "pose_strategy": "m38_2_partial_opening_cylinder",
                "eligible_count": 0,
                "selected_ring_instance_id": None,
                "robot_candidate": None,
                "instances": [],
            }

        overlay = _paint_fit_masks(rgb, fits)
        overlay = draw_overlay(overlay, instances, scene, intrinsics)
        cv2.imwrite(str(output / f"{capture_id}_m38_3_overlay.jpg"), overlay)
        write_json(
            output / f"{capture_id}_m38_3_result.json",
            {
                "stage": "M38.3_depth_partial_opening_constrained_cylinder",
                "capture_id": capture_id,
                "association_debug": association_debug,
                "depth_inference": inference_records,
                "preselected": [
                    {
                        "ring_instance_id": int(item["ring"].instance_id),
                        "mouth_instance_id": int(item["mouth"].instance_id),
                        "source": item["source"],
                        "evidence_score": item["score"],
                    }
                    for item in evidence_rows
                ],
                "fits": [
                    {
                        key: _strip_debug(value)
                        for key, value in fit.items()
                        if key not in {"_debug", "synthetic_mouth_instance", "synthetic_ring_instance"}
                    }
                    for fit in fits
                ],
                "scene": _strip_debug(scene),
                "timing_ms": {
                    "depth_inference_wall_ms": inference_wall_ms,
                    "fit_wall_ms": fit_wall_ms,
                    "geometry_wall_ms": geometry_wall_ms,
                    "total_wall_ms": inference_wall_ms + fit_wall_ms + geometry_wall_ms,
                },
            },
        )
        captures.append({
            "capture_id": capture_id,
            "segmented_match_count": len(matches),
            "depth_inference_eligible_count": sum(int(row["eligible"]) for row in inference_records),
            "preselected_ring_instance_ids": [int(item["ring"].instance_id) for item in evidence_rows],
            "local_fit_eligible_count": len(pose_by_ring),
            "final_eligible_count": scene.get("eligible_count"),
            "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
            "selected_clock_hour": scene.get("selected_clock_hour"),
            "depth_inference_wall_ms": inference_wall_ms,
            "fit_wall_ms": fit_wall_ms,
            "geometry_wall_ms": geometry_wall_ms,
            "total_wall_ms": inference_wall_ms + fit_wall_ms + geometry_wall_ms,
        })

    csv_path = output / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else ["capture_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "stage": "M38.3_depth_partial_opening_constrained_cylinder",
            "config": str(args.config.resolve()),
            "captures": captures,
            "fits": rows,
        },
    )
    print(json.dumps({
        "stage": "M38.3_depth_partial_opening_constrained_cylinder",
        "capture_count": len(captures),
        "fit_count": len(rows),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
