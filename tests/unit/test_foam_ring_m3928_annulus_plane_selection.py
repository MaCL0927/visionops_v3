from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    _box_reference_axes_camera,
    _m3928_select_annulus_plane,
    _robot_grasp_frame,
    _vector_angle_deg,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (
    _load_box_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LINE = REPO_ROOT / "production/foam_ring_grasp/config/line.yaml"
CLOCK3 = REPO_ROOT / "production/foam_ring_grasp/config/line_clock3_calibration.yaml"
BOX = REPO_ROOT / "production/foam_ring_grasp/config/box_model.json"


def _config(*, canonical: bool = False) -> GeometryConfig:
    box = json.loads(BOX.read_text(encoding="utf-8"))
    raw = {
        "plane": {
            "ransac_iterations": 320,
            "inlier_threshold_mm": 5.0,
            "random_seed": 3401,
        },
        "box_wall": {
            "enabled": False,
            "model": "calibrated_3d_cuboid",
            "calibrated_model": box,
            "_resolved_calibration_file": str(BOX),
        },
        "annulus_plane_selection": {
            "enabled": True,
            "reference": "calibrated_box_floor",
            "sector_count": 16,
            "inlier_threshold_mm": 5.0,
            "minimum_occupied_sectors": 6,
            "minimum_good_sector_coverage_deg": 135.0,
            "minimum_sector_inlier_ratio": 0.55,
            "maximum_sector_residual_p90_mm": 6.0,
            "measured_required_score_advantage_mm": 1.0,
            "measured_required_score_ratio": 0.82,
            "floor_parallel_rescue_enabled": True,
        },
        "flat_ring_normal_stabilization": {"enabled": False},
        "grasp_frame_orientation": {
            "mode": "canonical_clock_camera_axes" if canonical else "measured",
            "clock_hours": [3],
        },
        "robot_interface": {
            "camera_frame_id": "camera_color_optical_frame",
            "length_unit": "mm",
        },
    }
    return GeometryConfig(raw)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(helper - normal * float(np.dot(helper, normal)))
    v = _unit(np.cross(normal, u))
    return u, v


def _rotate(normal: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _unit(axis)
    theta = math.radians(angle_deg)
    kx, ky, kz = axis
    k = np.asarray([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    rot = np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    return _unit(rot @ normal)


def _annulus(
    normal: np.ndarray,
    *,
    seed: int,
    dense_bad_sector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = _unit(normal)
    u, v = _basis(normal)
    center = np.asarray([-50.0, 20.0, 760.0], dtype=np.float64)
    points = []
    pixels = []
    sectors = 16
    for sector in range(sectors):
        count = 500 if dense_bad_sector and sector == 0 else 12
        for j in range(count):
            theta = 2.0 * math.pi * (sector + (j + 0.5) / count) / sectors
            radius = 48.0 + rng.normal(0.0, 1.0)
            point = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
            point += normal * rng.normal(0.0, 0.45)
            if dense_bad_sector and sector == 0:
                point += normal * (5.0 + 0.10 * (j % 40))
            points.append(point)
            pixels.append([320.0 + 80.0 * math.cos(theta), 320.0 + 80.0 * math.sin(theta)])
    return np.asarray(points), np.rint(np.asarray(pixels)).astype(np.int32)


def test_m3929_production_config_supersedes_dual_hypothesis_with_floor_constrained_surface() -> None:
    raw = load_yaml(LINE)
    selector = raw["annulus_plane_selection"]
    assert selector["enabled"] is False
    assert selector["sector_count"] == 16
    assert selector["floor_parallel_rescue_enabled"] is True
    assert "preserve_measured_at_reference_disagreement_deg" not in selector
    front = raw["branch_a_front_surface"]
    assert front["enabled"] is True
    assert front["mode"] == "floor_constrained"
    assert front["reference"] == "calibrated_box_floor"
    assert front["height_estimation"]["minimum_good_sectors"] == 4
    assert raw["flat_ring_normal_stabilization"]["enabled"] is False
    assert raw["grasp_frame_orientation"]["mode"] == "measured"

    tf = np.asarray(
        raw["robot_pose_transform"]["hand_tcp_to_flange"]["T_hand_tcp_flange_rows"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(tf[:3, 3], [-164.567832, -3.149675, 35.546126], atol=1e-6)


def test_m3928_sector_balancing_rejects_dense_local_crescent_as_tilt_evidence() -> None:
    config = _config()
    _x_right, _y_down, z_inside = _box_reference_axes_camera(config)
    floor = -z_inside
    points, pixels = _annulus(floor, seed=1, dense_bad_sector=True)
    plane, diag = _m3928_select_annulus_plane(points, pixels, (320.0, 320.0), config)
    assert plane is not None
    assert diag["selected_hypothesis"] == "floor_parallel"
    assert diag["floor_parallel_rescue_applied"] is False
    assert _vector_angle_deg(plane.normal, floor) < 0.2
    expected_center = np.asarray([-50.0, 20.0, 760.0], dtype=np.float64)
    expected_offset = -float(np.dot(floor, expected_center))
    # Raw point-count offset is pulled by the 500-point bad crescent; the final
    # M39.2.8 equal-sector offset must stay on the physical ring plane.
    raw_point_weighted_offset = float(np.median(-(points @ floor)))
    assert abs(raw_point_weighted_offset - expected_offset) > 2.0
    assert abs(float(plane.offset) - expected_offset) < 0.8
    counts = diag["measured"]["sector_point_counts"]
    assert max(counts.values()) > 20 * min(counts.values())


def test_m3928_true_tilt_kept_only_when_measured_plane_materially_better() -> None:
    config = _config()
    x_right, _y_down, z_inside = _box_reference_axes_camera(config)
    floor = -z_inside
    tilted = _rotate(floor, x_right, 20.0)
    points, pixels = _annulus(tilted, seed=2)
    plane, diag = _m3928_select_annulus_plane(points, pixels, (320.0, 320.0), config)
    assert plane is not None
    assert diag["selected_hypothesis"] == "measured"
    assert diag["selection_reason"] in {
        "measured_materially_better_than_floor_parallel",
        "floor_parallel_not_supported_by_annulus",
    }
    assert _vector_angle_deg(plane.normal, tilted) < 1.0
    # Disagreement is diagnostic only; there is no 12-degree hard switch.
    assert diag["measured_vs_floor_normal_deg"] > 12.0


def test_m3928_floor_parallel_rescues_measured_fit_failure() -> None:
    config = _config()
    _x_right, _y_down, z_inside = _box_reference_axes_camera(config)
    floor = -z_inside
    points, pixels = _annulus(floor, seed=3)
    module = "production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry"
    with patch(
        module + "._m3928_sector_balanced_measured_plane",
        return_value=(None, {"status": "synthetic_measured_failure"}),
    ):
        plane, diag = _m3928_select_annulus_plane(points, pixels, (320.0, 320.0), config)
    assert plane is not None
    assert diag["selected_hypothesis"] == "floor_parallel"
    assert diag["floor_parallel_rescue_applied"] is True
    assert diag["selected_acceptable"] is True


def test_m3928_clock3_canonical_orientation_still_preserves_final_xyz() -> None:
    config = _config(canonical=True)
    candidate = {
        "clock_hour": 3,
        "image_angle_deg_from_positive_x": 0.0,
        "grasp_center_camera_mm": [-55.0, 20.0, 758.0],
        "closing_axis_camera": [0.99, 0.0, 0.14],
        "approach_vector_camera": [-0.14, 0.0, 0.99],
    }
    frame = _robot_grasp_frame(candidate, config)
    assert frame["orientation_policy"] == "canonical_clock_camera_axes"
    np.testing.assert_allclose(frame["origin_camera_mm"], candidate["grasp_center_camera_mm"], atol=1e-12)
    np.testing.assert_allclose(frame["x_closing_axis_camera"], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame["z_approach_axis_camera"], [0.0, 0.0, 1.0], atol=1e-9)


def test_m3928_clock3_loads_box_reference_even_when_collision_is_disabled() -> None:
    raw = copy.deepcopy(load_yaml(CLOCK3))
    assert raw["box_wall"]["enabled"] is False
    assert raw["annulus_plane_selection"]["enabled"] is False
    assert raw["branch_a_front_surface"]["enabled"] is True
    assert "calibrated_model" not in raw["box_wall"]
    _load_box_calibration(raw, CLOCK3)
    assert "calibrated_model" in raw["box_wall"]
    assert raw["box_wall"]["_resolved_calibration_file"].endswith("box_model.json")
