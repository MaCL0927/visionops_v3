#!/usr/bin/env python3
"""Replay saved RGB-D bundles through the M38.5 A/B/D/C policy.

M38.5 branch D returns pure-side outer-contact geometry only. It intentionally
skips inner-finger and complete-gripper checks. M36/M37.6 stay disabled in the
production configuration, while branch C remains the terminal rejection path.
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/m38_5_replay"))
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
        branch_d = scene.get("m38_5_branch_d") if isinstance(scene.get("m38_5_branch_d"), Mapping) else {}
        branch_b = scene.get("m38_3_branch_b") if isinstance(scene.get("m38_3_branch_b"), Mapping) else {}
        hybrid = scene.get("hybrid_grasp") if isinstance(scene.get("hybrid_grasp"), Mapping) else {}
        timing = hybrid.get("timing_ms") if isinstance(hybrid.get("timing_ms"), Mapping) else {}
        candidate = branch_d.get("candidate") if isinstance(branch_d.get("candidate"), Mapping) else {}
        outer = candidate.get("outer_contact") if isinstance(candidate.get("outer_contact"), Mapping) else {}
        quality = candidate.get("quality") if isinstance(candidate.get("quality"), Mapping) else {}
        row = {
            "capture_id": capture_id,
            "selected_grasp_branch": scene.get("selected_grasp_branch"),
            "target_found": bool(scene.get("robot_candidate") is not None),
            "robot_ready": bool((scene.get("robot_candidate") or {}).get("robot_ready", False)) if isinstance(scene.get("robot_candidate"), Mapping) else False,
            "terminal_reject": bool(branch_c.get("fast_terminated", False)),
            "terminal_reason": branch_c.get("decision"),
            "rings_detected": scene.get("rings_detected"),
            "mouths_detected": scene.get("mouths_detected"),
            "m38_3_fast_gate_ring_ids": json.dumps(branch_b.get("fast_gate_skipped_ring_ids") or []),
            "m38_5_attempt_count": branch_d.get("attempt_count"),
            "m38_5_selected_ring_instance_id": branch_d.get("selected_ring_instance_id"),
            "m38_5_radial_inlier_ratio": quality.get("radial_inlier_ratio"),
            "m38_5_radial_residual_median_mm": quality.get("radial_residual_median_mm"),
            "m38_5_radial_residual_p90_mm": quality.get("radial_residual_p90_mm"),
            "m38_5_axis_view_angle_deg": quality.get("axis_view_angle_deg"),
            "m38_5_contact_support_error_mm": outer.get("support_error_mm"),
            "m38_5_outer_contact_ms": timing.get("m38_5_outer_contact_ms"),
            "m38_3_rim_pinch_geometry_ms": timing.get("m38_3_branch_b_geometry_ms"),
            "old_m38_4_geometry_ms": _old_geometry_ms(bundle),
            "m38_5_replay_wall_ms": wall_ms,
        }
        rows.append(row)
        overlay = draw_overlay(rgb, instances, scene, intrinsics)
        cv2.putText(
            overlay,
            f"M38.5 {row['selected_grasp_branch']} {wall_ms:.0f} ms",
            (10, max(24, overlay.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output / f"{capture_id}_m38_5_overlay.jpg"), overlay)
        write_json(output / f"{capture_id}_m38_5_result.json", _strip_debug(scene))

    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["capture_id"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"stage": "M38.5_pure_side_outer_contact", "rows": rows})
    print(json.dumps({"stage": "M38.5", "captures": len(rows), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
