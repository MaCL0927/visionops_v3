#!/usr/bin/env python3
"""carton_tube_check 业务 Mock HTTP 服务。"""

from __future__ import annotations

import argparse
from typing import Sequence

from edge.gateway_adapter.apps.common.app_service_base import add_common_arguments, run_business_app
from .decision import evaluate
from .mock_inputs import MOCK_CASES
from .register_map import decision_register_values, make_register_map


DEFAULT_CONFIG = {
    "schema_version": "1.0", "kind": "app",
    "app": {"name": "carton_tube_check", "version": "1.0"},
    "rules": {
        "target_class_names": ["carton_tube", "paper_tube", "tube"], "score_threshold": 0.5,
        "allow_multi_target": False, "roi_xyxy": [0, 0, 1280, 720],
        "expected_center_xy": [640, 360], "center_tolerance_px": 250,
        "min_bbox_width": 80, "max_bbox_width": 300, "min_bbox_height": 80,
        "max_bbox_height": 300, "register_base": 100,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps carton_tube_check 业务 Mock")
    add_common_arguments(parser, default_port=19110, default_modbus_port=1510, mock_cases=MOCK_CASES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_business_app(build_parser().parse_args(argv), defaults=DEFAULT_CONFIG, mock_factories=MOCK_CASES, decide=evaluate, definition_factory=make_register_map, register_values=decision_register_values)


if __name__ == "__main__":
    raise SystemExit(main())
