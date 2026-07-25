"""使用小型 fake tensor 验证 C++ YOLO 后处理契约。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.interfaces.validate_interface_examples import validate_example


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def postprocess_fixture_binary(shared_postprocess_fixture_binary: Path) -> Path:
    return shared_postprocess_fixture_binary

@pytest.mark.parametrize(
    ("fixture_task", "expected_task"),
    [
        ("detection", "detection"),
        ("detection_split", "detection"),
        ("obb", "obb"),
        ("obb_rockchip", "obb"),
        ("segmentation", "segmentation"),
    ],
)
def test_postprocess_fixture_matches_inference_contract(
    postprocess_fixture_binary: Path, fixture_task: str, expected_task: str
) -> None:
    completed = subprocess.run(
        [str(postprocess_fixture_binary), fixture_task],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    validate_example(result, f"{fixture_task} postprocess fixture")
    assert result["task_type"] == expected_task
    assert result["timing"]["total_ms"] == 3
    assert result["detections"]
    if fixture_task in {"obb", "obb_rockchip"}:
        assert len(result["detections"][0]["obb"]["points"]) == 4
    if fixture_task == "segmentation":
        assert result["detections"][0]["mask"]["encoding"] == "polygon"


def test_detection_roi_filters_target_after_full_frame_postprocess(
    postprocess_fixture_binary: Path,
) -> None:
    completed = subprocess.run(
        [str(postprocess_fixture_binary), "detection_roi"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["task_type"] == "detection"
    assert result["detections"] == []


def test_detection_coordinates_are_restored_to_full_frame_after_input_roi(
    postprocess_fixture_binary: Path,
) -> None:
    completed = subprocess.run(
        [str(postprocess_fixture_binary), "detection_input_roi"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    detection = result["detections"][0]
    assert detection["center_xy"] == pytest.approx([1063.5, 604.4227], abs=1e-3)
    assert detection["bbox_xyxy"] == pytest.approx(
        [1010.125, 564.3914, 1116.875, 644.4539], abs=1e-3
    )


def test_obb_points_and_angle_are_restored_after_input_roi(
    postprocess_fixture_binary: Path,
) -> None:
    completed = subprocess.run(
        [str(postprocess_fixture_binary), "obb_input_roi"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    obb = result["detections"][0]["obb"]
    assert [obb["cx"], obb["cy"]] == pytest.approx([1063.5, 604.4227], abs=1e-3)
    assert obb["angle_deg"] == pytest.approx(14.3239, abs=1e-3)
    assert len(obb["points"]) == 4
    assert all(850 <= point[0] <= 1277 for point in obb["points"])
    assert all(490 <= point[1] <= 719 for point in obb["points"])
