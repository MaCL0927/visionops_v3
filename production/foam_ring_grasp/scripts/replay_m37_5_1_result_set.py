#!/usr/bin/env python3
"""Replay saved debug bundles with M37.5.1 staged pose-validation selection."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import _bundle_input, _strip_debug
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import _load_box_calibration


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved bundles with M37.6")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("production/foam_ring_grasp/config/line.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/m37_5_1_replay"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _selected_fit(scene: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ring_id = scene.get("selected_ring_instance_id")
    for fit in ((scene.get("side_ring_branch") or {}).get("fits") or []):
        if fit.get("ring_instance_id") == ring_id:
            return fit
    return None


def main() -> int:
    args = _args()
    config_path = args.config.resolve()
    raw_config = load_yaml(config_path)
    _load_box_calibration(raw_config, config_path)
    geometry_config = GeometryConfig(dict(raw_config))
    bundles = sorted(
        [path for path in args.root.resolve().iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    if args.limit > 0:
        bundles = bundles[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        capture_id, _rgb, depth, intrinsics, instances, _inputs = _bundle_input(bundle)
        started = time.perf_counter()
        scene = run_hybrid_grasp(
            instances,
            depth,
            intrinsics,
            raw_config=raw_config,
            geometry_config=geometry_config,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        hybrid = scene.get("hybrid_grasp") if isinstance(scene.get("hybrid_grasp"), Mapping) else {}
        side = scene.get("side_ring_branch") if isinstance(scene.get("side_ring_branch"), Mapping) else {}
        layering = scene.get("depth_layering") if isinstance(scene.get("depth_layering"), Mapping) else {}
        safety = scene.get("m37_5_pose_safety") if isinstance(scene.get("m37_5_pose_safety"), Mapping) else {}
        selected_fit = _selected_fit(scene)
        row = {
            "capture_id": capture_id,
            "selected_branch": hybrid.get("selected_branch"),
            "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
            "selected_depth_layer_index": layering.get("selected_layer_index"),
            "selected_surface_depth_mm": layering.get("selected_surface_depth_mm"),
            "m36_pose_conflict_rejections": safety.get("m36_pose_conflict_rejection_count"),
            "m37_uncertainty_rejections": safety.get("m37_uncertainty_rejection_count"),
            "m37_screen_attempt_count": side.get("screen_attempt_count", side.get("fast_attempt_count")),
            "m37_final_validation_count": side.get("final_validation_count"),
            "m37_local_refinement_count": side.get("accurate_refinement_count"),
            "selected_fit_score": selected_fit.get("fit_score") if selected_fit else None,
            "selected_normal_inlier_ratio": selected_fit.get("normal_inlier_ratio") if selected_fit else None,
            "selected_normal_axis_median_deg": selected_fit.get("normal_axis_median_deg") if selected_fit else None,
            "selected_normal_radial_median_deg": selected_fit.get("normal_radial_median_deg") if selected_fit else None,
            "selected_bootstrap_spread_deg": (
                (((selected_fit.get("pose_uncertainty") or {}).get("bootstrap") or {}).get("maximum_axis_spread_deg"))
                if selected_fit else None
            ),
            "target_found": hybrid.get("target_found"),
            "hybrid_ms": (hybrid.get("timing_ms") or {}).get("total_ms"),
            "wall_ms": wall_ms,
        }
        rows.append(row)
        write_json(args.output / f"{capture_id}_scene.json", _strip_debug(scene))
        print(json.dumps(row, ensure_ascii=False))
    write_json(args.output / "summary.json", {"stage": "M37.6", "count": len(rows), "rows": rows})
    if rows:
        with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
