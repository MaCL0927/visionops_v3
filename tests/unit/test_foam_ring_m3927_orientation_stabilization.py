from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    PlaneModel,
    _box_reference_axes_camera,
    _robot_grasp_frame,
    _stabilize_flat_ring_pose_plane,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (
    _load_box_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LINE = REPO_ROOT / "production/foam_ring_grasp/config/line.yaml"
CLOCK3 = REPO_ROOT / "production/foam_ring_grasp/config/line_clock3_calibration.yaml"
BOX = REPO_ROOT / "production/foam_ring_grasp/config/box_model.json"


def _config(*, stabilization: bool = True, canonical: bool = False) -> GeometryConfig:
    box = json.loads(BOX.read_text(encoding="utf-8"))
    raw = {
        "box_wall": {
            "enabled": True,
            "model": "calibrated_3d_cuboid",
            "calibrated_model": box,
            "_resolved_calibration_file": str(BOX),
        },
        "flat_ring_normal_stabilization": {
            "enabled": stabilization,
            "reference": "calibrated_box_floor",
            "require_reference": True,
            "pose_strategies": ["m38_1_front_annulus"],
            "snap_max_reference_disagreement_deg": 9.0,
            "preserve_measured_at_reference_disagreement_deg": 12.0,
            "maximum_ellipse_tilt_for_stabilization_deg": 90.0,
            "minimum_plane_inlier_ratio": 0.40,
            "maximum_plane_residual_p95_mm": 7.0,
            "reanchor_inlier_threshold_mm": 7.0,
        },
        "grasp_frame_orientation": {
            "mode": "canonical_clock_camera_axes" if canonical else "measured",
            "clock_hours": [3],
            "require_box_reference": True,
        },
        "robot_interface": {
            "camera_frame_id": "camera_color_optical_frame",
            "length_unit": "mm",
        },
    }
    return GeometryConfig(raw)


def _plane_with_normal(normal: np.ndarray) -> tuple[PlaneModel, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    center = np.asarray([-80.0, 25.0, 756.0], dtype=np.float64)
    # Build deterministic support in a 60x60 mm patch on the requested plane.
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    points = []
    for a in np.linspace(-30.0, 30.0, 9):
        for b in np.linspace(-30.0, 30.0, 9):
            points.append(center + a * u + b * v)
    points = np.asarray(points, dtype=np.float64)
    return (
        PlaneModel(
            normal=normal,
            offset=-float(np.dot(normal, center)),
            centroid=center,
            inlier_mask=np.ones(len(points), dtype=bool),
            inlier_ratio=0.90,
            residual_median_mm=1.0,
            residual_p95_mm=3.0,
        ),
        points,
    )


def _ellipse(tilt_deg: float = 5.0) -> dict:
    major = 40.0
    minor = major * math.cos(math.radians(tilt_deg))
    return {
        "center_uv": (320.0, 320.0),
        "major_px": major,
        "minor_px": minor,
        "angle_deg": 0.0,
    }


def test_m3927_production_config_enables_flat_stabilization_and_corrected_tool_tf() -> None:
    raw = load_yaml(LINE)
    stab = raw["flat_ring_normal_stabilization"]
    assert stab["enabled"] is True
    assert stab["snap_max_reference_disagreement_deg"] == pytest.approx(9.0)
    assert stab["preserve_measured_at_reference_disagreement_deg"] == pytest.approx(12.0)
    assert raw["grasp_frame_orientation"]["mode"] == "measured"
    tf = np.asarray(
        raw["robot_pose_transform"]["hand_tcp_to_flange"]["T_hand_tcp_flange_rows"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(tf[:3, 3], [-164.567832, -3.149675, 35.546126], atol=1e-6)
    np.testing.assert_allclose(
        tf[:3, :3],
        [
            [0.995921716, -0.087619482, 0.021511897],
            [-0.089118123, -0.992545706, 0.083132319],
            [0.014067531, -0.084710381, -0.996306306],
        ],
        atol=1e-9,
    )


def test_m3927_near_flat_raw_normal_snaps_to_calibrated_box_floor() -> None:
    config = _config(stabilization=True)
    x_right, _y_down, z_inside = _box_reference_axes_camera(config)
    reference = -z_inside
    theta = math.radians(8.0)
    raw = math.cos(theta) * reference + math.sin(theta) * x_right
    plane, points = _plane_with_normal(raw)
    stable, diag = _stabilize_flat_ring_pose_plane(
        plane,
        points,
        _ellipse(5.0),
        {"fx": 607.56, "fy": 607.31, "cx": 320.0, "cy": 320.0},
        config,
        "m38_1_front_annulus",
    )
    assert diag["applied"] is True
    assert diag["mode"] == "snap_to_box_floor"
    assert diag["reference_weight"] == pytest.approx(1.0)
    np.testing.assert_allclose(stable.normal, reference, atol=1e-9)


def test_m3927_real_tilt_above_preserve_threshold_is_not_flattened() -> None:
    config = _config(stabilization=True)
    x_right, _y_down, z_inside = _box_reference_axes_camera(config)
    reference = -z_inside
    theta = math.radians(18.0)
    raw = math.cos(theta) * reference + math.sin(theta) * x_right
    plane, points = _plane_with_normal(raw)
    stable, diag = _stabilize_flat_ring_pose_plane(
        plane,
        points,
        _ellipse(18.0),
        {"fx": 607.56, "fy": 607.31, "cx": 320.0, "cy": 320.0},
        config,
        "m38_1_front_annulus",
    )
    assert diag["applied"] is False
    assert diag["mode"] == "preserve_measured"
    np.testing.assert_allclose(stable.normal, plane.normal, atol=1e-12)


def test_m3927_clock3_canonical_orientation_uses_box_axes_but_preserves_xyz() -> None:
    config = _config(stabilization=False, canonical=True)
    candidate = {
        "clock_hour": 3,
        "image_angle_deg_from_positive_x": 0.0,
        "grasp_center_camera_mm": [-55.0, 20.0, 758.0],
        # Deliberately noisy 8-degree-ish measured axes; canonical mode must ignore them.
        "closing_axis_camera": [0.99, 0.0, 0.14],
        "approach_vector_camera": [-0.14, 0.0, 0.99],
    }
    frame = _robot_grasp_frame(candidate, config)
    assert frame["orientation_policy"] == "canonical_clock_camera_axes"
    assert frame["orientation_diagnostics"]["applied"] is True
    np.testing.assert_allclose(frame["origin_camera_mm"], candidate["grasp_center_camera_mm"], atol=1e-12)
    np.testing.assert_allclose(frame["x_closing_axis_camera"], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame["z_approach_axis_camera"], [0.0, 0.0, 1.0], atol=1e-9)
    transform = np.asarray(frame["T_camera_grasp_rows"], dtype=np.float64)
    np.testing.assert_allclose(transform[:3, 3], candidate["grasp_center_camera_mm"], atol=1e-12)


def test_m3927_clock3_camera_canonical_does_not_require_box_collision_model() -> None:
    raw = copy.deepcopy(load_yaml(CLOCK3))
    assert raw["box_wall"]["enabled"] is False
    assert raw["grasp_frame_orientation"]["mode"] == "canonical_clock_camera_axes"
    assert "calibrated_model" not in raw["box_wall"]
    _load_box_calibration(raw, CLOCK3)
    # Camera-axis canonical calibration deliberately remains independent of the
    # box model; this lets collision stay disabled without losing orientation.
    assert "calibrated_model" not in raw["box_wall"]
