#!/usr/bin/env python3
"""Summarize M37.5.1 staged pose-validation timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _latest(root: Path) -> Path:
    candidates = sorted(
        root.glob("*/online_geometry_result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no online_geometry_result.json below {root}")
    return candidates[0]


def _number(value: Any) -> str:
    try:
        return f"{float(value):10.3f}"
    except (TypeError, ValueError):
        return "         -"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize M37.5.1 hybrid timing")
    parser.add_argument("result", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=Path("data/foam_ring_online_geometry"))
    args = parser.parse_args()
    path = args.result or _latest(args.root)
    document = json.loads(path.read_text(encoding="utf-8"))
    scene = document.get("scene") if isinstance(document.get("scene"), Mapping) else {}
    hybrid = scene.get("hybrid_grasp") if isinstance(scene.get("hybrid_grasp"), Mapping) else {}
    side = scene.get("side_ring_branch") if isinstance(scene.get("side_ring_branch"), Mapping) else {}
    layering = scene.get("depth_layering") if isinstance(scene.get("depth_layering"), Mapping) else {}
    timing = hybrid.get("timing_ms") if isinstance(hybrid.get("timing_ms"), Mapping) else {}
    safety = scene.get("m37_5_pose_safety") if isinstance(scene.get("m37_5_pose_safety"), Mapping) else {}
    processor_timing = document.get("timing_ms") if isinstance(document.get("timing_ms"), Mapping) else {}

    print(f"result: {path.resolve()}")
    print(
        "branch: selected={}, layer={}, depth_mm={}, rank={}, target_found={}".format(
            hybrid.get("selected_branch") or scene.get("selected_grasp_branch") or "none",
            layering.get("selected_layer_index"),
            layering.get("selected_surface_depth_mm"),
            layering.get("selected_depth_rank"),
            hybrid.get("target_found"),
        )
    )
    print(
        "M37: screen_attempts={}, final_validations={}, local_refines={}, selected={}".format(
            side.get("screen_attempt_count", side.get("fast_attempt_count")),
            side.get("final_validation_count"),
            side.get("accurate_refinement_count"),
            side.get("selected_ring_instance_id"),
        )
    )
    print(
        "Pose safety: M36_conflicts={}, M37_uncertain={}, normal_constrained={}".format(
            safety.get("m36_pose_conflict_rejection_count"),
            safety.get("m37_uncertainty_rejection_count"),
            safety.get("normal_constrained_axis_enabled"),
        )
    )
    selected_fit = None
    for fit in side.get("fits") or []:
        if fit.get("ring_instance_id") == side.get("selected_ring_instance_id"):
            selected_fit = fit
            break
    if isinstance(selected_fit, Mapping):
        uncertainty = selected_fit.get("pose_uncertainty") if isinstance(selected_fit.get("pose_uncertainty"), Mapping) else {}
        bootstrap = uncertainty.get("bootstrap") if isinstance(uncertainty.get("bootstrap"), Mapping) else {}
        print(
            "Selected M37 normal evidence: inlier={:.3f}, axis_med={:.2f}deg, radial_med={:.2f}deg, bootstrap_max={:.2f}deg".format(
                float(selected_fit.get("normal_inlier_ratio") or 0.0),
                float(selected_fit.get("normal_axis_median_deg") or 0.0),
                float(selected_fit.get("normal_radial_median_deg") or 0.0),
                float(bootstrap.get("maximum_axis_spread_deg") or 0.0),
            )
        )

    print("\nDepth layers")
    for layer in layering.get("layers") or []:
        print(
            "  L{} anchor={} max={} candidates={} ids={}".format(
                layer.get("layer_index"),
                layer.get("anchor_depth_mm"),
                layer.get("maximum_depth_mm"),
                layer.get("candidate_count"),
                layer.get("ring_instance_ids"),
            )
        )
    print("\nHybrid branch timing (ms)")
    for key in (
        "association_prepass_ms",
        "depth_preselection_ms",
        "depth_layer_build_ms",
        "m36_branch_ms",
        "m37_lightweight_preselection_ms",
        "m37_screen_total_ms",
        "m37_final_validation_ms",
        "m37_fast_total_ms",
        "m37_local_accurate_ms",
        "m37_evaluated_instance_total_ms",
        "total_ms",
    ):
        print(f"  {key:40s} {_number(timing.get(key))}")
    print("\nEnd-to-end processor timing (ms)")
    for key in (
        "runtime_http_ms",
        "runtime_json_decode_ms",
        "exact_rgbd_match_ms",
        "polygon_to_mask_ms",
        "prepare_total_ms",
        "geometry_ms",
        "visualization_ms",
        "save_outputs_ms",
        "total_ms",
    ):
        print(f"  {key:40s} {_number(processor_timing.get(key))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
