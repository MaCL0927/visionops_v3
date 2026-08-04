#!/usr/bin/env python3
"""Print a compact timing summary for one M37.2 result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data" / "foam_ring_side_template_m37"


def _latest_result(root: Path) -> Path:
    candidates = sorted(
        root.glob("*/side_ring_template_result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到M37结果: {root}")
    return candidates[0]


def _value(mapping: Mapping[str, Any], key: str) -> float:
    try:
        return float(mapping.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总M37.2分阶段耗时")
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result_path = (
        args.result.expanduser().resolve()
        if args.result is not None
        else _latest_result(args.root.expanduser().resolve())
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    timing = payload.get("timing_ms") or {}
    execution = payload.get("execution") or {}
    selected = payload.get("selected") or {}
    selected_timing = selected.get("timing_ms") or {}
    axis = selected_timing.get("axis_search") or {}
    fast = axis.get("fast") or {}
    accurate = axis.get("accurate") or {}

    print(f"result: {result_path}")
    print(
        "execution: mode={mode}, profile={profile}, candidates={candidates}, "
        "evaluated={evaluated}, deferred={deferred}, selected={selected}".format(
            mode=execution.get("mode"),
            profile=execution.get("search_profile"),
            candidates=payload.get("candidate_count"),
            evaluated=payload.get("evaluated_count"),
            deferred=payload.get("deferred_count"),
            selected=payload.get("selected_ring_instance_id"),
        )
    )
    print("\nScene timing (ms)")
    for key in (
        "association_ms",
        "candidate_filter_sort_ms",
        "fit_loop_ms",
        "output_generation_ms",
        "evaluated_instance_total_ms",
        "total_ms",
    ):
        print(f"  {key:32s} {_value(timing, key):10.3f}")

    if selected:
        print("\nSelected instance timing (ms)")
        for key in (
            "point_extraction_ms",
            "mask_prepare_ms",
            "depth_deproject_ms",
            "depth_trim_ms",
            "axis_template_fit_ms",
            "endpoint_and_grasp_ms",
            "quality_gate_ms",
            "total_ms",
        ):
            print(f"  {key:32s} {_value(selected_timing, key):10.3f}")
        print("\nAxis search")
        print(f"  requested_profile: {axis.get('requested_profile')}")
        print(f"  final_profile:     {axis.get('final_profile')}")
        print(f"  fallback_used:     {axis.get('fallback_used')}")
        print(f"  fallback_reasons:  {axis.get('fallback_reasons')}")
        if fast:
            print(
                "  fast: global={:.3f} ms, local={:.3f} ms, final={:.3f} ms, "
                "candidates={}".format(
                    _value(fast, "global_search_ms"),
                    _value(fast, "local_refine_ms"),
                    _value(fast, "final_full_point_evaluation_ms"),
                    int(fast.get("candidate_evaluations", 0)),
                )
            )
        if accurate:
            print(
                "  accurate fallback: total={:.3f} ms, candidates={}".format(
                    _value(accurate, "total_ms"),
                    int(accurate.get("candidate_evaluations", 0)),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
