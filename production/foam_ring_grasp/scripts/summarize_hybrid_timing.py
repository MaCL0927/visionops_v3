#!/usr/bin/env python3
"""Summarize M37.3 hybrid branch and timing from a service/geometry JSON."""
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
    parser = argparse.ArgumentParser(description="Summarize M37.3 hybrid branch timing")
    parser.add_argument("result", type=Path, nargs="?")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/foam_ring_online_geometry"),
    )
    args = parser.parse_args()
    path = args.result or _latest(args.root)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("result root must be an object")
    scene = document.get("scene") if isinstance(document.get("scene"), Mapping) else {}
    hybrid = scene.get("hybrid_grasp") if isinstance(scene.get("hybrid_grasp"), Mapping) else {}
    side = scene.get("side_ring_branch") if isinstance(scene.get("side_ring_branch"), Mapping) else {}
    timing = hybrid.get("timing_ms") if isinstance(hybrid.get("timing_ms"), Mapping) else {}
    processor_timing = document.get("timing_ms") if isinstance(document.get("timing_ms"), Mapping) else {}

    print(f"result: {path.resolve()}")
    print(
        "branch: selected={}, fallback={}, target_found={}".format(
            hybrid.get("selected_branch") or scene.get("selected_grasp_branch") or "none",
            hybrid.get("fallback_triggered"),
            hybrid.get("target_found"),
        )
    )
    print(
        "M36: matched_pairs={}, candidate_found={}".format(
            scene.get("matched_pairs"), hybrid.get("m36_candidate_found")
        )
    )
    print(
        "M37: candidates={}, evaluated={}, deferred={}, selected={}".format(
            side.get("candidate_count"),
            side.get("evaluated_count"),
            side.get("deferred_count"),
            side.get("selected_ring_instance_id"),
        )
    )
    print("\nHybrid branch timing (ms)")
    for key in (
        "m36_branch_ms",
        "m37_candidate_filter_sort_ms",
        "m37_fit_loop_ms",
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
