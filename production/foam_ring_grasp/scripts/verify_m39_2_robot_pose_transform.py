#!/usr/bin/env python3
"""Validate the M39.2 LEFT robot-pose transform configuration.

Without --candidate-json this verifies the locked left-hand hand-eye calibration and
the static-tool-transform safety gates. With --candidate-json it converts one saved
candidate/result into base_link Visual Grasp/pregrasp poses and, only when exact
static transforms are configured, hand_tcp / left_link7 poses.
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

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml, write_json  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.robot_pose_transform import (  # noqa: E402
    RobotPoseTransformer,
)

DEFAULT_CONFIG = REPO_ROOT / "production" / "foam_ring_grasp" / "config" / "line.yaml"


def _candidate_from_document(document: Any) -> Mapping[str, Any] | None:
    if not isinstance(document, Mapping):
        return None
    if isinstance(document.get("grasp_frame_camera"), Mapping):
        return document
    candidate = document.get("candidate")
    if isinstance(candidate, Mapping):
        return candidate
    scene = document.get("scene")
    if isinstance(scene, Mapping) and isinstance(scene.get("robot_candidate"), Mapping):
        return scene["robot_candidate"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M39.2 LEFT: verify camera -> base -> Visual Grasp -> gated hand_tcp / left_link7 chain"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    transformer = RobotPoseTransformer.from_mapping(load_yaml(config_path), config_path)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.2_left_robot_pose_transform_verify",
        "status": "passed",
        "transformer": transformer.status(),
    }

    if args.candidate_json is not None:
        source = args.candidate_json.expanduser().resolve()
        document = json.loads(source.read_text(encoding="utf-8"))
        candidate = _candidate_from_document(document)
        report["candidate_source"] = str(source)
        report["robot_pose_transform"] = transformer.transform_candidate(candidate)

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
