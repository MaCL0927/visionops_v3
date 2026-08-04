#!/usr/bin/env python3
"""Summarize M36.4.2 geometry timing from one saved online result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _latest_result(root: Path) -> Path:
    candidates = sorted(root.glob("*/online_geometry_result.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no online_geometry_result.json under {root}")
    return candidates[-1]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _print_aggregate(title: str, aggregate: Mapping[str, Any]) -> None:
    rows = []
    for name, payload in aggregate.items():
        data = _mapping(payload)
        rows.append((float(data.get("total_ms") or 0.0), str(name), data))
    rows.sort(reverse=True)
    print(f"\n{title}")
    for total, name, data in rows:
        print(
            f"  {name:34s} total={total:9.3f} ms  "
            f"mean={float(data.get('mean_ms') or 0.0):8.3f} ms  "
            f"max={float(data.get('max_ms') or 0.0):8.3f} ms  "
            f"count={int(float(data.get('count') or 0))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize M36.4.2 first-valid/adaptive-clock geometry timing")
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/opt/visionops_v3/data/foam_ring_online_geometry"),
    )
    args = parser.parse_args()
    path = args.result or _latest_result(args.root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    scene = _mapping(payload.get("scene"))
    optimization = _mapping(scene.get("geometry_optimization"))
    timing = _mapping(payload.get("timing_ms"))
    scene_timing = _mapping(scene.get("timing_ms"))
    detail = _mapping(scene.get("timing_detail"))

    print(f"result: {path}")
    print(f"capture_timestamp_ms: {payload.get('capture_timestamp_ms')}")
    print(f"mode: {optimization.get('mode')}")
    print(
        "pairs: matched=%s, analyzed=%s, deferred=%s, first_valid_rank=%s, early_exit=%s"
        % (
            optimization.get("matched_pair_count"),
            optimization.get("fully_analyzed_pair_count"),
            optimization.get("deferred_pair_count"),
            optimization.get("first_valid_pair_rank"),
            optimization.get("early_exit_triggered"),
        )
    )
    print(
        "candidates: light=%s (primary=%s fallback=%s), full=%s, valid_full=%s, fallback_used=%s"
        % (
            optimization.get("light_candidate_count"),
            optimization.get("primary_light_candidate_count"),
            optimization.get("fallback_light_candidate_count"),
            optimization.get("full_candidate_evaluated_count"),
            optimization.get("full_candidate_valid_count"),
            optimization.get("adaptive_fallback_used"),
        )
    )
    print(
        "selected: ring=%s, clock=%s, eligible=%s"
        % (
            scene.get("selected_ring_instance_id"),
            scene.get("selected_clock_hour"),
            scene.get("eligible_count"),
        )
    )
    print(
        "top-level: runtime=%s ms, polygon=%s ms, geometry=%s ms, total=%s ms"
        % (
            _mapping(_mapping(payload.get("runtime")).get("timing")).get("total_ms"),
            timing.get("polygon_to_mask_ms"),
            timing.get("geometry_ms"),
            timing.get("total_ms"),
        )
    )
    print("\nScene timing")
    for name, value in sorted(scene_timing.items(), key=lambda item: float(item[1]), reverse=True):
        print(f"  {name:34s} {float(value):9.3f} ms")
    _print_aggregate("Pair timing aggregate", _mapping(detail.get("pairs")))
    _print_aggregate("Full-candidate timing aggregate", _mapping(detail.get("full_candidates")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
