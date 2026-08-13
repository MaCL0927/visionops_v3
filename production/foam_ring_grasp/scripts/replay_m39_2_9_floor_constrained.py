#!/usr/bin/env python3
"""Replay RGB-D debug bundles with the M39.2.9 Branch-A front-surface model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    _box_reference_axes_camera,
    _vector_angle_deg,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import _load_box_calibration, _strip_debug  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import _bundle_input  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="foam_ring_online_geometry directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("production/foam_ring_grasp/config/line.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/m39_2_9_floor_constrained_replay"))
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Also fail a row when the complete clock/collision pipeline has no robot_candidate.",
    )
    return parser.parse_args()


def _selected_or_first_branch_a(scene: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ring_id = scene.get("selected_ring_instance_id")
    if ring_id is not None:
        for item in scene.get("instances") or []:
            if isinstance(item, Mapping) and int(item.get("ring_instance_id", -1)) == int(ring_id):
                return item
    for item in scene.get("instances") or []:
        if isinstance(item, Mapping) and isinstance(item.get("m38_branch_a"), Mapping):
            return item
    return None


def main() -> int:
    args = _args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    raw = load_yaml(config_path)
    _load_box_calibration(raw, config_path)
    geometry = GeometryConfig(dict(raw))
    axes = _box_reference_axes_camera(geometry)
    if axes is None:
        raise RuntimeError("M39.2.9 replay requires calibrated box reference")
    _x_right, _y_down, z_inside = axes
    floor_normal = -np.asarray(z_inside, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for bundle in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            capture_id, _rgb, depth, intrinsics, instances, _inputs = _bundle_input(bundle)
            scene = run_hybrid_grasp(
                instances,
                depth,
                intrinsics,
                raw_config=raw,
                geometry_config=geometry,
            )
            item = _selected_or_first_branch_a(scene)
            branch_a = item.get("m38_branch_a") if isinstance(item, Mapping) else {}
            branch_a = branch_a if isinstance(branch_a, Mapping) else {}
            surface = branch_a.get("front_surface") if isinstance(branch_a.get("front_surface"), Mapping) else {}
            layer = branch_a.get("front_depth_layer") if isinstance(branch_a.get("front_depth_layer"), Mapping) else {}
            measured = surface.get("measured_plane_diagnostic") if isinstance(surface.get("measured_plane_diagnostic"), Mapping) else {}
            pose = item.get("pose") if isinstance(item, Mapping) and isinstance(item.get("pose"), Mapping) else {}
            final_normal = pose.get("normal_toward_camera")
            final_vs_floor = None
            if final_normal is not None:
                final_vs_floor = _vector_angle_deg(np.asarray(final_normal, dtype=np.float64), floor_normal)
            row_failures: list[str] = []
            if not item:
                row_failures.append("branch_a_instance_missing")
            if str(surface.get("status") or "") != "ok":
                row_failures.append("front_surface_not_ok")
            if final_vs_floor is None or final_vs_floor > 0.05:
                row_failures.append("final_normal_not_floor_constrained")
            if args.require_target and scene.get("robot_candidate") is None:
                row_failures.append("robot_candidate_missing")
            row = {
                "capture_id": capture_id,
                "target_found": bool(scene.get("robot_candidate") is not None),
                "selected_grasp_branch": scene.get("selected_grasp_branch"),
                "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
                "selected_clock_hour": scene.get("selected_clock_hour"),
                "final_tilt_deg": item.get("tilt_deg") if isinstance(item, Mapping) else None,
                "raw_measured_tilt_deg": item.get("raw_tilt_deg") if isinstance(item, Mapping) else None,
                "final_normal_vs_floor_deg": final_vs_floor,
                "front_layer_applied": bool(layer.get("applied", False)),
                "front_layer_center_gap_mm": layer.get("center_gap_mm"),
                "front_layer_near_fraction": layer.get("near_fraction"),
                "front_surface_status": surface.get("status"),
                "front_height_source": surface.get("height_source"),
                "good_sector_count": surface.get("good_sector_count"),
                "rejected_sectors": surface.get("rejected_sectors"),
                "measured_vs_floor_deg": measured.get("measured_vs_floor_deg"),
                "measured_normal_status": measured.get("status"),
                "plane_centroid_camera_mm": (item.get("plane") or {}).get("centroid_camera_mm") if isinstance(item, Mapping) else None,
                "rejection_reasons": item.get("rejection_reasons") if isinstance(item, Mapping) else None,
                "passed": not row_failures,
                "failures": row_failures,
            }
            rows.append(row)
            failures.extend(f"{capture_id}: {reason}" for reason in row_failures)
            write_json(output / f"{capture_id}_scene.json", _strip_debug(scene))
        except Exception as error:  # replay should continue across damaged bundles
            failures.append(f"{bundle.name}: {error}")
            rows.append({
                "capture_id": bundle.name,
                "passed": False,
                "failures": [f"exception: {error}"],
            })

    report = {
        "schema_version": "1.0",
        "stage": "M39.2.9_floor_constrained_front_surface_replay",
        "status": "passed" if not failures else "failed",
        "bundle_count": len(rows),
        "target_found_count": sum(bool(row.get("target_found")) for row in rows),
        "front_surface_pass_count": sum(bool(row.get("passed")) for row in rows),
        "maximum_final_normal_vs_floor_deg": max(
            (float(row["final_normal_vs_floor_deg"]) for row in rows if row.get("final_normal_vs_floor_deg") is not None),
            default=None,
        ),
        "maximum_measured_vs_floor_deg": max(
            (float(row["measured_vs_floor_deg"]) for row in rows if row.get("measured_vs_floor_deg") is not None),
            default=None,
        ),
        "rows": rows,
        "failures": failures,
    }
    write_json(output / "replay_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
