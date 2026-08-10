#!/usr/bin/env python3
"""Replay the M39.2 problem-frame bundle and verify the M39.2.2 geometry fix.

The regression protects three behaviors:
1. the previously correct stacked-ring frame remains unchanged;
2. strong two-depth-layer single-ring frames select the camera-near front layer;
3. the 157.5-degree inlier-coverage boundary frame is no longer falsely rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (  # noqa: E402
    _load_box_calibration,
    _strip_debug,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import _bundle_input  # noqa: E402


EXPECTED: dict[str, dict[str, Any]] = {
    "1786175478063": {
        "max_tilt_deg": 30.0,
        "selected_ring_instance_id": 6,
        "selected_clock_hour": 11,
        "front_layer_applied": False,
    },
    "1786179505407": {"max_tilt_deg": 15.0, "front_layer_applied": True},
    "1786179530831": {"max_tilt_deg": 15.0, "front_layer_applied": True},
    "1786179570210": {"max_tilt_deg": 15.0, "front_layer_applied": True},
    "1786321107227": {"max_tilt_deg": 20.0, "front_layer_applied": True},
    "1786321549452": {
        "max_tilt_deg": 15.0,
        "front_layer_applied": False,
        "minimum_inlier_coverage_deg": 150.0,
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="foam_ring_online_geometry directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("production/foam_ring_grasp/config/line.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/m39_2_2_geometry_regression"))
    return parser.parse_args()


def _selected_instance(scene: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ring_id = scene.get("selected_ring_instance_id")
    if ring_id is None:
        return None
    for item in scene.get("instances") or []:
        if isinstance(item, Mapping) and int(item.get("ring_instance_id", -1)) == int(ring_id):
            return item
    return None


def main() -> int:
    args = _args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    raw = load_yaml(config_path)
    _load_box_calibration(raw, config_path)
    geometry = GeometryConfig(dict(raw))
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for capture_id, expected in EXPECTED.items():
        bundle = root / capture_id
        if not bundle.is_dir():
            failures.append(f"{capture_id}: bundle_missing")
            continue
        _capture, _rgb, depth, intrinsics, instances, _ = _bundle_input(bundle)
        scene = run_hybrid_grasp(
            instances,
            depth,
            intrinsics,
            raw_config=raw,
            geometry_config=geometry,
        )
        selected = _selected_instance(scene)
        diag = (
            selected.get("m38_branch_a")
            if isinstance(selected, Mapping) and isinstance(selected.get("m38_branch_a"), Mapping)
            else {}
        )
        layer = diag.get("front_depth_layer") if isinstance(diag.get("front_depth_layer"), Mapping) else {}
        tilt = float(selected.get("tilt_deg", 999.0)) if isinstance(selected, Mapping) else 999.0
        row = {
            "capture_id": capture_id,
            "target_found": bool(scene.get("robot_candidate") is not None),
            "selected_grasp_branch": scene.get("selected_grasp_branch"),
            "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
            "selected_clock_hour": scene.get("selected_clock_hour"),
            "tilt_deg": tilt,
            "front_layer_applied": bool(layer.get("applied", False)),
            "front_layer_depth_gap_mm": layer.get("depth_gap_mm"),
            "plane_z_mm": ((selected.get("plane") or {}).get("centroid_camera_mm") or [None, None, None])[2]
            if isinstance(selected, Mapping)
            else None,
            "inlier_angular_coverage_deg": diag.get("inlier_angular_coverage_deg"),
            "rejection_reasons": selected.get("rejection_reasons") if isinstance(selected, Mapping) else None,
        }
        row_failures: list[str] = []
        if not row["target_found"]:
            row_failures.append("target_not_found")
        if tilt > float(expected["max_tilt_deg"]):
            row_failures.append(f"tilt_above_{expected['max_tilt_deg']}")
        if bool(row["front_layer_applied"]) != bool(expected["front_layer_applied"]):
            row_failures.append("front_layer_policy_mismatch")
        if "selected_ring_instance_id" in expected and row["selected_ring_instance_id"] != expected["selected_ring_instance_id"]:
            row_failures.append("selected_ring_changed")
        if "selected_clock_hour" in expected and row["selected_clock_hour"] != expected["selected_clock_hour"]:
            row_failures.append("selected_clock_changed")
        minimum_coverage = expected.get("minimum_inlier_coverage_deg")
        if minimum_coverage is not None and float(row["inlier_angular_coverage_deg"] or 0.0) < float(minimum_coverage):
            row_failures.append("inlier_coverage_below_expected")
        row["passed"] = not row_failures
        row["failures"] = row_failures
        rows.append(row)
        failures.extend(f"{capture_id}: {reason}" for reason in row_failures)
        write_json(output / f"{capture_id}_scene.json", _strip_debug(scene))

    report = {
        "schema_version": "1.0",
        "stage": "M39.2.2_geometry_fix_regression",
        "status": "passed" if not failures else "failed",
        "rows": rows,
        "failures": failures,
    }
    write_json(output / "regression_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
