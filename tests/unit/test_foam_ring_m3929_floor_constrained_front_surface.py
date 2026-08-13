from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    _box_reference_axes_camera,
    _m3929_floor_constrained_front_surface,
    _m3929_select_front_depth_layer,
    _vector_angle_deg,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BOX = REPO_ROOT / "production/foam_ring_grasp/config/box_model.json"


def _config() -> GeometryConfig:
    box = json.loads(BOX.read_text(encoding="utf-8"))
    return GeometryConfig(
        {
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
            "branch_a_front_surface": {
                "enabled": True,
                "mode": "floor_constrained",
                "reference": "calibrated_box_floor",
                "require_reference": True,
                "depth_layer": {
                    "kmeans_iterations": 24,
                    "minimum_center_gap_mm": 24.0,
                    "minimum_layer_fraction": 0.08,
                    "minimum_near_points": 40,
                },
                "height_estimation": {
                    "sector_count": 16,
                    "minimum_sector_points": 5,
                    "minimum_good_sectors": 4,
                    "preferred_good_sectors": 6,
                    "minimum_height_outlier_threshold_mm": 5.0,
                    "maximum_height_outlier_threshold_mm": 20.0,
                    "mad_scale": 3.0,
                    "point_fallback_enabled": True,
                    "minimum_point_fallback_count": 40,
                    "point_trim_low_percentile": 10,
                    "point_trim_high_percentile": 90,
                    "plane_inlier_threshold_mm": 8.0,
                },
                "measured_plane_diagnostic": {
                    "enabled": True,
                    "warning_disagreement_deg": 10.0,
                    "severe_disagreement_deg": 20.0,
                },
            },
        }
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(helper - normal * float(np.dot(helper, normal)))
    v = _unit(np.cross(normal, u))
    return u, v


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _unit(axis)
    theta = math.radians(angle_deg)
    kx, ky, kz = axis
    k = np.asarray([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    rot = np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    return _unit(rot @ vector)


def _ring_points(
    normal: np.ndarray,
    center: np.ndarray,
    *,
    seed: int,
    points_per_sector: int = 12,
    sector_bias_mm: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = _unit(normal)
    u, v = _basis(normal)
    points = []
    pixels = []
    for sector in range(16):
        bias = float((sector_bias_mm or {}).get(sector, 0.0))
        for j in range(points_per_sector):
            theta = 2.0 * math.pi * (sector + (j + 0.5) / points_per_sector) / 16.0
            radius = 46.0 + rng.normal(0.0, 0.7)
            point = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
            point += normal * (bias + rng.normal(0.0, 0.5))
            points.append(point)
            pixels.append([320.0 + 80.0 * math.cos(theta), 320.0 + 80.0 * math.sin(theta)])
    return np.asarray(points), np.rint(np.asarray(pixels)).astype(np.int32)


def test_m3929_two_cluster_depth_layer_handles_transition_tail() -> None:
    config = _config()
    _x, _y, z_inside = _box_reference_axes_camera(config)
    floor_normal = -z_inside
    center = np.asarray([-50.0, 20.0, 755.0], dtype=np.float64)
    front, front_px = _ring_points(floor_normal, center, seed=1, points_per_sector=10)
    far = front + 67.0 * z_inside
    far_px = front_px.copy()
    # Add a transition tail that makes the sorted-depth largest-gap test less clean.
    transition = np.vstack([front[:20] + offset * z_inside for offset in (18.0, 30.0, 42.0, 54.0)])
    transition_px = np.vstack([front_px[:20] for _ in range(4)])
    points = np.vstack([front, far, transition])
    pixels = np.vstack([front_px, far_px, transition_px])

    selected, selected_px, diag = _m3929_select_front_depth_layer(points, pixels, config)
    assert diag["applied"] is True
    assert diag["center_gap_mm"] > 40.0
    assert len(selected) >= 120
    assert len(selected) == len(selected_px)
    # Selected layer should remain close to the true front coordinate.
    selected_inside = selected @ z_inside
    true_inside = float(np.median(front @ z_inside))
    assert abs(float(np.median(selected_inside)) - true_inside) < 5.0


def test_m3929_bad_measured_normal_is_diagnostic_only() -> None:
    config = _config()
    x_right, _y, z_inside = _box_reference_axes_camera(config)
    floor_normal = -z_inside
    tilted = _rotate(floor_normal, x_right, 25.0)
    center = np.asarray([-40.0, 15.0, 755.0], dtype=np.float64)
    points, pixels = _ring_points(tilted, center, seed=2, points_per_sector=14)

    plane, diag = _m3929_floor_constrained_front_surface(points, pixels, (320.0, 320.0), config)
    assert plane is not None
    assert _vector_angle_deg(plane.normal, floor_normal) < 1e-6
    measured = diag["measured_plane_diagnostic"]
    assert measured["available"] is True
    assert measured["measured_vs_floor_deg"] > 20.0
    assert measured["status"] == "depth_normal_severely_unreliable"
    assert diag["normal_source"] == "calibrated_box_floor"


def test_m3929_sector_height_outliers_do_not_move_final_plane() -> None:
    config = _config()
    _x, _y, z_inside = _box_reference_axes_camera(config)
    floor_normal = -z_inside
    center = np.asarray([-55.0, 10.0, 755.0], dtype=np.float64)
    points, pixels = _ring_points(
        floor_normal,
        center,
        seed=3,
        points_per_sector=12,
        sector_bias_mm={0: -45.0, 1: -38.0, 8: 35.0},
    )
    plane, diag = _m3929_floor_constrained_front_surface(points, pixels, (320.0, 320.0), config)
    assert plane is not None
    assert _vector_angle_deg(plane.normal, floor_normal) < 1e-6
    expected_offset = -float(np.dot(floor_normal, center))
    assert abs(float(plane.offset) - expected_offset) < 2.0
    assert diag["good_sector_count"] >= 10
    assert len(diag["rejected_sectors"]) >= 2
    assert diag["sector_height_outlier_threshold_mm"] <= 20.0
