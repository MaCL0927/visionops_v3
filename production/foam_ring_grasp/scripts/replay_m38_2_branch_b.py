#!/usr/bin/env python3
"""Replay saved RGB-D bundles through M38.2 branch B only.

The diagnostic keeps only matched ``foam_ring + ring_mouth`` instances.  It
fits the observed local outer-cylinder side patch and partial axial end, then
passes eligible fits through the unchanged rim-pinch and collision evaluator.
M38.1, M36 and M37.6 are intentionally skipped so branch-B geometry can be
inspected independently.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.partial_opening_cylinder import (  # noqa: E402
    fit_partial_opening_cylinder,
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/m38_2_branch_b_replay"))
    parser.add_argument("--capture", action="append", default=[])
    return parser.parse_args()


def _row(capture_id: str, fit: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = fit.get("diagnostics") if isinstance(fit.get("diagnostics"), Mapping) else {}
    timing = fit.get("timing_ms") if isinstance(fit.get("timing_ms"), Mapping) else {}
    return {
        "capture_id": capture_id,
        "ring_instance_id": fit.get("ring_instance_id"),
        "mouth_instance_id": fit.get("mouth_instance_id"),
        "eligible_local_fit": bool(fit.get("eligible", False)),
        "radial_inlier_ratio": diagnostics.get("radial_inlier_ratio"),
        "radial_residual_median_mm": diagnostics.get("radial_residual_median_mm"),
        "radial_residual_p90_mm": diagnostics.get("radial_residual_p90_mm"),
        "normal_axis_median_deg": diagnostics.get("normal_axis_median_deg"),
        "normal_axis_p90_deg": diagnostics.get("normal_axis_p90_deg"),
        "visible_normal_span_deg": diagnostics.get("visible_normal_span_deg"),
        "projected_axis_error_deg": diagnostics.get("projected_axis_error_deg"),
        "observed_axis_span_mm": diagnostics.get("observed_axis_span_mm"),
        "axis_view_angle_deg": diagnostics.get("axis_view_angle_deg"),
        "endpoint_support_point_count": diagnostics.get("endpoint_support_point_count"),
        "endpoint_plane_inlier_ratio": diagnostics.get("endpoint_plane_inlier_ratio"),
        "endpoint_axial_residual_p90_mm": diagnostics.get("endpoint_axial_residual_p90_mm"),
        "opening_center_error_px": diagnostics.get("opening_center_error_px"),
        "partial_mouth_overlap_ratio": diagnostics.get("partial_mouth_overlap_ratio"),
        "opening_near_margin_mm": diagnostics.get("opening_near_margin_mm"),
        "candidate_axis_count": diagnostics.get("candidate_axis_count"),
        "fit_total_ms": timing.get("total_ms"),
        "rejection_reasons": ";".join(str(value) for value in fit.get("rejection_reasons") or []),
    }


def _paint_fit_masks(image: np.ndarray, fits: List[Mapping[str, Any]]) -> np.ndarray:
    output = image.copy()
    for fit in fits:
        debug = fit.get("_debug") if isinstance(fit.get("_debug"), Mapping) else {}
        side = debug.get("side_surface_mask")
        endpoint = debug.get("endpoint_support_mask")
        synthetic = debug.get("synthetic_mouth_mask")
        synthetic_outer = debug.get("synthetic_outer_mask")
        if isinstance(side, np.ndarray) and side.shape == output.shape[:2]:
            layer = output.copy()
            layer[side.astype(bool)] = (255, 0, 255)
            output = cv2.addWeighted(layer, 0.25, output, 0.75, 0.0)
        if isinstance(endpoint, np.ndarray) and endpoint.shape == output.shape[:2]:
            contours, _ = cv2.findContours(endpoint.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
        if (
            isinstance(synthetic_outer, np.ndarray)
            and synthetic_outer.shape == output.shape[:2]
            and np.any(synthetic_outer)
        ):
            contours, _ = cv2.findContours(
                synthetic_outer.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        if isinstance(synthetic, np.ndarray) and synthetic.shape == output.shape[:2] and np.any(synthetic):
            contours, _ = cv2.findContours(synthetic.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    return output


def main() -> int:
    args = _parse_args()
    raw = load_yaml(args.config.resolve())
    geometry_config = GeometryConfig(dict(raw))
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
        matches, _, _, association_debug = _associate_ring_mouths_detailed(
            rings, mouths, geometry_config
        )
        fits: List[Dict[str, Any]] = []
        pose_by_ring: Dict[int, Mapping[str, Any]] = {}
        replacements: Dict[int, SegmentationInstance] = {}
        ring_replacements: Dict[int, SegmentationInstance] = {}
        fit_started = time.perf_counter()
        for ring, mouth, metrics in matches:
            fit = fit_partial_opening_cylinder(
                ring, mouth, metrics, rings, depth, intrinsics, raw
            )
            fits.append(fit)
            rows.append(_row(capture_id, fit))
            payload = fit.get("pose_payload")
            synthetic = fit.get("synthetic_mouth_instance")
            synthetic_ring = fit.get("synthetic_ring_instance")
            if (
                bool(fit.get("eligible"))
                and isinstance(payload, Mapping)
                and isinstance(synthetic, SegmentationInstance)
                and isinstance(synthetic_ring, SegmentationInstance)
            ):
                pose_by_ring[int(ring.instance_id)] = dict(payload)
                replacements[int(synthetic.instance_id)] = synthetic
                ring_replacements[int(synthetic_ring.instance_id)] = synthetic_ring
        fit_wall_ms = (time.perf_counter() - fit_started) * 1000.0

        geometry_wall_ms = 0.0
        if pose_by_ring:
            branch_instances = _replace_ring_instances(instances, ring_replacements)
            branch_instances = _replace_mouth_instances(branch_instances, replacements)
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
        cv2.imwrite(str(output / f"{capture_id}_m38_2_overlay.jpg"), overlay)
        write_json(
            output / f"{capture_id}_m38_2_result.json",
            {
                "stage": "M38.2_branch_B_partial_opening_local_cylinder",
                "capture_id": capture_id,
                "association_debug": association_debug,
                "fits": [
                    {
                        key: _strip_debug(value)
                        for key, value in fit.items()
                        if key
                        not in {
                            "_debug",
                            "synthetic_mouth_instance",
                            "synthetic_ring_instance",
                        }
                    }
                    for fit in fits
                ],
                "scene": _strip_debug(scene),
                "timing_ms": {
                    "fit_wall_ms": fit_wall_ms,
                    "geometry_wall_ms": geometry_wall_ms,
                    "total_wall_ms": fit_wall_ms + geometry_wall_ms,
                },
            },
        )
        captures.append(
            {
                "capture_id": capture_id,
                "matched_pair_count": len(matches),
                "local_fit_eligible_count": len(pose_by_ring),
                "final_eligible_count": scene.get("eligible_count"),
                "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
                "selected_clock_hour": scene.get("selected_clock_hour"),
                "fit_wall_ms": fit_wall_ms,
                "geometry_wall_ms": geometry_wall_ms,
                "total_wall_ms": fit_wall_ms + geometry_wall_ms,
            }
        )

    csv_path = output / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else ["capture_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "stage": "M38.2_branch_B_partial_opening_local_cylinder",
            "config": str(args.config.resolve()),
            "captures": captures,
            "pairs": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "M38.2_branch_B_partial_opening_local_cylinder",
                "capture_count": len(captures),
                "pair_count": len(rows),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
