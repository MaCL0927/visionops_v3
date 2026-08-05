#!/usr/bin/env python3
"""Replay saved online RGB-D bundles through M38.1 branch A only.

This diagnostic intentionally skips M36 and M37.6.  It measures whether a
matched ring/mouth pair supplies enough directly observed 3-D front-annulus
support, then runs the existing rim-pinch and collision evaluation unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    _associate_ring_mouths_detailed,
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (  # noqa: E402
    _scoped_m38a_geometry_config,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import (  # noqa: E402
    load_yaml,
    write_json,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (  # noqa: E402
    _strip_debug,
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/m38_1_branch_a_replay"))
    parser.add_argument("--capture", action="append", default=[])
    return parser.parse_args()


def _pair_row(capture_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    annulus = item.get("m38_branch_a") if isinstance(item.get("m38_branch_a"), dict) else {}
    plane = item.get("plane") if isinstance(item.get("plane"), dict) else {}
    return {
        "capture_id": capture_id,
        "ring_instance_id": item.get("ring_instance_id"),
        "mouth_instance_id": item.get("mouth_instance_id"),
        "eligible": bool(item.get("eligible", False)),
        "opening_clear": bool(annulus.get("opening_clear", False)),
        "tilt_deg": item.get("tilt_deg"),
        "annulus_point_count": annulus.get("annulus_point_count"),
        "annulus_depth_valid_ratio": annulus.get("annulus_depth_valid_ratio"),
        "angular_coverage_deg": annulus.get("angular_coverage_deg"),
        "inlier_angular_coverage_deg": annulus.get("inlier_angular_coverage_deg"),
        "plane_inlier_ratio": plane.get("inlier_ratio"),
        "plane_residual_p95_mm": plane.get("residual_p95_mm"),
        "rejection_reasons": ";".join(str(value) for value in item.get("rejection_reasons") or []),
    }


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    raw = load_yaml(config_path)
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
        matches, _, _, _ = _associate_ring_mouths_detailed(
            rings,
            mouths,
            geometry_config,
        )
        matched_ring_ids = [int(ring.instance_id) for ring, _mouth, _metrics in matches]
        started = time.perf_counter()
        scene = analyze_scene(
            instances,
            depth,
            intrinsics,
            _scoped_m38a_geometry_config(geometry_config, matched_ring_ids),
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        for item in scene.get("instances") or []:
            if isinstance(item, dict) and item.get("m38_branch_a") is not None:
                rows.append(_pair_row(capture_id, item))
        overlay = draw_overlay(rgb, instances, scene, intrinsics)
        cv2.imwrite(str(output / f"{capture_id}_m38_1_overlay.jpg"), overlay)
        write_json(output / f"{capture_id}_m38_1_scene.json", _strip_debug(scene))
        captures.append(
            {
                "capture_id": capture_id,
                "matched_ring_ids": matched_ring_ids,
                "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
                "eligible_count": scene.get("eligible_count"),
                "wall_ms": wall_ms,
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
            "stage": "M38.1_branch_A_front_annulus",
            "config": str(config_path),
            "captures": captures,
            "pairs": rows,
        },
    )
    print(
        json.dumps(
            {
                "stage": "M38.1_branch_A_front_annulus",
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
