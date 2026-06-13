"""接口示例数据的轻量契约测试。"""

from __future__ import annotations

from pathlib import Path

from tools.interfaces.dump_interface_summary import build_summary
from tools.interfaces.validate_interface_examples import validate_directories


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "interfaces/schemas"
EXAMPLE_DIR = PROJECT_ROOT / "interfaces/examples"


def test_all_interface_examples_pass_lightweight_validation() -> None:
    examples = validate_directories(SCHEMA_DIR, EXAMPLE_DIR)
    assert len(examples) == 12


def test_interface_summary_contains_expected_types() -> None:
    summary = build_summary(SCHEMA_DIR, EXAMPLE_DIR)
    assert summary["message_types"] == {
        "camera_frame": 1,
        "gateway_message": 5,
        "inference_result": 4,
        "model_package_manifest": 1,
        "runtime_status": 1,
    }
    assert summary["task_types"] == {
        "detection": 2,
        "obb": 1,
        "roi_classification": 1,
        "segmentation": 1,
    }
