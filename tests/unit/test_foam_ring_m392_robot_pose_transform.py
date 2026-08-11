from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.robot_pose_transform import (
    RobotPoseTransformError,
    RobotPoseTransformer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "production" / "foam_ring_grasp" / "config" / "line.yaml"
CALIB = REPO_ROOT / "production" / "foam_ring_grasp" / "config" / "handeye_left_20260810_190310_robot_default_base.json"


def _candidate() -> dict:
    return {
        "grasp_branch": "m36_mouth_visible_rim_pinch",
        "pregrasp_center_camera_mm": [10.0, 20.0, -70.0],
        "grasp_frame_camera": {
            "coordinate_frame": "camera_color_optical_frame",
            "length_unit": "mm",
            "T_camera_grasp_rows": [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }


def test_left_calibration_is_locked_to_24sample_park_robot_default_base() -> None:
    raw = load_yaml(CONFIG)
    transformer = RobotPoseTransformer.from_mapping(raw, CONFIG)
    status = transformer.status()
    assert status["arm"] == "left"
    assert status["handeye"]["robot_arm"] == "left"
    assert status["handeye"]["selected_method"] == "PARK"
    assert status["handeye"]["quality_status"] == "PASS"
    assert status["handeye"]["sample_count_used"] == 24
    np.testing.assert_allclose(
        transformer.T_base_camera_mm[:3, 3],
        [427.395220, 200.122131, 1192.767193],
        atol=1e-9,
    )


def test_m392_left_camera_to_base_grasp_is_ready_but_tcp_is_gated_without_exact_origin() -> None:
    raw = copy.deepcopy(load_yaml(CONFIG))
    raw["robot_pose_transform"]["visual_grasp_to_hand_tcp"]["enabled"] = False
    raw["robot_pose_transform"]["visual_grasp_to_hand_tcp"]["T_grasp_hand_tcp_rows"] = None
    raw["robot_pose_transform"]["hand_tcp_to_flange"]["enabled"] = False
    raw["robot_pose_transform"]["hand_tcp_to_flange"]["T_hand_tcp_flange_rows"] = None
    transformer = RobotPoseTransformer.from_mapping(raw, CONFIG)
    result = transformer.transform_candidate(_candidate())
    assert result["status"] == "base_grasp_ready"
    assert result["arm"] == "left"
    assert result["camera_frame_alias_match"] is True
    assert result["hand_tcp"]["available"] is False
    assert result["flange"]["available"] is False

    Tbc = transformer.T_base_camera_mm
    Tcg = np.asarray(_candidate()["grasp_frame_camera"]["T_camera_grasp_rows"])
    expected_grasp = Tbc @ Tcg
    np.testing.assert_allclose(
        np.asarray(result["grasp"]["T_base_grasp_mm"]), expected_grasp, atol=1e-9
    )

    Tpre = Tcg.copy()
    Tpre[:3, 3] = [10.0, 20.0, -70.0]
    expected_pre = Tbc @ Tpre
    np.testing.assert_allclose(
        np.asarray(result["pregrasp"]["T_base_pregrasp_mm"]), expected_pre, atol=1e-9
    )
    quaternion = np.asarray(result["grasp"]["visual_grasp_pose_base"]["quaternion_xyzw"])
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-9)


def test_m392_exact_tool_transform_enables_left_hand_tcp_chain() -> None:
    raw = copy.deepcopy(load_yaml(CONFIG))
    visual = raw["robot_pose_transform"]["visual_grasp_to_hand_tcp"]
    visual["enabled"] = True
    visual["origin_policy"] = "test_exact_static_transform"
    visual["T_grasp_hand_tcp_rows"] = [
        [0.0, 0.0, 1.0, 12.0],
        [0.0, -1.0, 0.0, -3.0],
        [1.0, 0.0, 0.0, 25.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transformer = RobotPoseTransformer.from_mapping(raw, CONFIG)
    result = transformer.transform_candidate(_candidate())
    assert result["status"] == "ok"
    assert result["hand_tcp"]["available"] is True

    Tbc = transformer.T_base_camera_mm
    Tcg = np.asarray(_candidate()["grasp_frame_camera"]["T_camera_grasp_rows"])
    Tgh = np.asarray(visual["T_grasp_hand_tcp_rows"], dtype=float)
    expected_grasp = Tbc @ Tcg @ Tgh
    np.testing.assert_allclose(
        np.asarray(result["grasp"]["T_base_hand_tcp_mm"]), expected_grasp, atol=1e-9
    )


def test_branch_without_m38_grasp_frame_is_not_transformed() -> None:
    transformer = RobotPoseTransformer.from_mapping(load_yaml(CONFIG), CONFIG)
    result = transformer.transform_candidate(
        {"grasp_branch": "m38_6_pure_side_outer_contact"}
    )
    assert result["status"] == "not_applicable"
    assert result["reason"] == "candidate_has_no_grasp_frame_camera"


def test_wrong_camera_frame_is_rejected() -> None:
    transformer = RobotPoseTransformer.from_mapping(load_yaml(CONFIG), CONFIG)
    candidate = _candidate()
    candidate["grasp_frame_camera"]["coordinate_frame"] = "depth_camera"
    with pytest.raises(RobotPoseTransformError, match="camera frame"):
        transformer.transform_candidate(candidate)


def test_right_arm_calibration_is_hard_rejected(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_yaml(CONFIG))
    calibration = json.loads(CALIB.read_text(encoding="utf-8"))
    calibration["robot_arm"] = "right"
    bad = tmp_path / "bad_right_calibration.json"
    bad.write_text(json.dumps(calibration), encoding="utf-8")
    section = raw["robot_pose_transform"]["handeye"]
    section["calibration_file"] = str(bad)
    section["sha256"] = ""
    with pytest.raises(RobotPoseTransformError, match="机械臂不匹配"):
        RobotPoseTransformer.from_mapping(raw, CONFIG)


def test_flange_pose_is_never_guessed_when_fixed_transform_missing() -> None:
    raw = copy.deepcopy(load_yaml(CONFIG))
    raw["robot_pose_transform"]["hand_tcp_to_flange"]["enabled"] = False
    raw["robot_pose_transform"]["hand_tcp_to_flange"]["T_hand_tcp_flange_rows"] = None
    transformer = RobotPoseTransformer.from_mapping(raw, CONFIG)
    result = transformer.transform_candidate(_candidate())
    assert result["flange"]["available"] is False
    assert "collision-model" in result["flange"]["reason"]
    assert "T_base_left_link7_mm" not in result["flange"]
