#!/usr/bin/env python3
"""carton_partition_check 业务 Mock HTTP 服务。"""

from __future__ import annotations

import argparse
from typing import Sequence

from edge.gateway_adapter.apps.common.app_service_base import add_common_arguments, run_business_app
from .decision import evaluate
from .mock_inputs import MOCK_CASES
from .register_map import decision_register_values, make_register_map


DEFAULT_CONFIG = {
    "schema_version": "1.0", "kind": "app",
    "app": {"name": "carton_partition_check", "version": "1.0"},
    "rules": {
        "target_class_names": ["cell", "partition_cell", "slot"],
        "defect_class_names": ["missing_cell", "broken_partition", "foreign_body", "defect"],
        "score_threshold": 0.5, "defect_score_threshold": 0.5,
        "expected_cell_count": 12, "min_cell_count": 12, "max_cell_count": 12,
        "roi_xyxy": [0, 0, 1280, 720], "register_base": 200,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps carton_partition_check 业务 Mock")
    add_common_arguments(parser, default_port=19120, default_modbus_port=1520, mock_cases=MOCK_CASES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_business_app(build_parser().parse_args(argv), defaults=DEFAULT_CONFIG, mock_factories=MOCK_CASES, decide=evaluate, definition_factory=make_register_map, register_values=decision_register_values)


if __name__ == "__main__":
    raise SystemExit(main())
