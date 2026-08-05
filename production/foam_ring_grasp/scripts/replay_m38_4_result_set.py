#!/usr/bin/env python3
"""Replay saved RGB-D bundles through the full M38.4 A/B/C policy.

M36 and M37.6 remain in the repository but the production M38.4 configuration
keeps both disabled. When M38.1 and M38.3 produce no collision-checked candidate,
branch C returns an explicit terminal rejection and operator action.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import _strip_debug  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import _bundle_input  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.visualization import draw_overlay  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("production/foam_ring_grasp/config/line.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/m38_4_replay"))
    parser.add_argument("--capture", action="append", default=[])
    return parser.parse_args()


def _old_geometry_ms(bundle: Path) -> float | None:
    path = bundle / "online_geometry_result.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    timing = document.get("timing_ms") if isinstance(document.get("timing_ms"), Mapping) else {}
    value = timing.get("geometry_ms")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = _args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = load_yaml(args.config.resolve())
    geometry = GeometryConfig(dict(raw))
    selected = {str(value) for value in args.capture}
    rows: list[dict[str, Any]] = []

    for bundle in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected and bundle.name not in selected:
            continue
        capture_id, rgb, depth, intrinsics, instances, _ = _bundle_input(bundle)
        started = time.perf_counter()
        scene = run_hybrid_grasp(
            instances,
            depth,
            intrinsics,
            raw_config=raw,
            geometry_config=geometry,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        branch_c = scene.get("m38_4_branch_c") if isinstance(scene.get("m38_4_branch_c"), Mapping) else {}
        hybrid = scene.get("hybrid_grasp") if isinstance(scene.get("hybrid_grasp"), Mapping) else {}
        side = scene.get("side_ring_branch") if isinstance(scene.get("side_ring_branch"), Mapping) else {}
        row = {
            "capture_id": capture_id,
            "selected_grasp_branch": scene.get("selected_grasp_branch"),
            "target_found": bool(scene.get("robot_candidate") is not None),
            "terminal_reject": bool(branch_c.get("fast_terminated", False)),
            "terminal_reason": branch_c.get("decision"),
            "operator_action": branch_c.get("operator_action"),
            "rings_detected": scene.get("rings_detected"),
            "mouths_detected": scene.get("mouths_detected"),
            "m38_1_candidate_found": hybrid.get("m38_1_candidate_found"),
            "m38_3_candidate_found": hybrid.get("m38_3_candidate_found"),
            "m36_attempt_count": (scene.get("m37_5_timing") or {}).get("m36_attempt_count"),
            "m37_fast_attempt_count": side.get("fast_attempt_count"),
            "old_m38_3_geometry_ms": _old_geometry_ms(bundle),
            "m38_4_replay_wall_ms": wall_ms,
        }
        rows.append(row)
        overlay = draw_overlay(rgb, instances, scene, intrinsics)
        cv2.putText(
            overlay,
            f"M38.4 {row['selected_grasp_branch']} {wall_ms:.0f} ms",
            (10, max(24, overlay.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output / f"{capture_id}_m38_4_overlay.jpg"), overlay)
        write_json(output / f"{capture_id}_m38_4_result.json", _strip_debug(scene))

    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["capture_id"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"stage": "M38.4_branch_c_fast_reject", "rows": rows})
    print(json.dumps({"stage": "M38.4", "captures": len(rows), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
