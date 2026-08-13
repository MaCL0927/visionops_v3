#!/usr/bin/env python3
"""Synthetic regression checks for M39.2.8 annulus final-plane selection.

Covers the three geometry decisions that do not require a real RGB-D capture:
1) a flat ring with a locally dense biased crescent must not tilt the result;
2) a genuinely tilted annulus must beat the floor-parallel hypothesis;
3) if the measured hypothesis is unavailable, a supported floor plane rescues it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    _m3928_select_annulus_plane,
    _vector_angle_deg,
)

BOX_JSON = ROOT / "production/foam_ring_grasp/config/box_model.json"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _unit(np.asarray(axis, dtype=np.float64))
    a = math.radians(angle_deg)
    kx, ky, kz = axis
    k = np.asarray([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _config() -> GeometryConfig:
    box = json.loads(BOX_JSON.read_text(encoding="utf-8"))
    return GeometryConfig({
        "plane": {
            "ransac_iterations": 320,
            "inlier_threshold_mm": 5.0,
            "random_seed": 3401,
        },
        "box_wall": {
            "enabled": False,
            "model": "calibrated_3d_cuboid",
            "calibrated_model": box,
        },
        "annulus_plane_selection": {
            "enabled": True,
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
    })


def _box_floor_normal(config: GeometryConfig) -> np.ndarray:
    rotation = np.asarray(config.section("box_wall")["calibrated_model"]["rotation_camera_from_box_rows"])
    return _unit(-rotation[:, 2])


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0])
    u = _unit(helper - normal * float(np.dot(helper, normal)))
    v = _unit(np.cross(normal, u))
    return u, v


def _annulus(
    normal: np.ndarray,
    *,
    seed: int,
    per_sector: int = 12,
    dense_bad_sector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = _unit(normal)
    u, v = _basis(normal)
    center = np.asarray([0.0, 0.0, 720.0], dtype=np.float64)
    pts = []
    px = []
    sectors = 16
    for sector in range(sectors):
        count = 500 if dense_bad_sector and sector == 0 else per_sector
        for j in range(count):
            theta = 2.0 * math.pi * (sector + (j + 0.5) / count) / sectors
            radius = 48.0 + rng.normal(0.0, 1.5)
            point = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
            point += normal * rng.normal(0.0, 0.55)
            if dense_bad_sector and sector == 0:
                # Dense crescent receives a systematic depth ramp. Raw point-count
                # fitting is attracted to it; one-sector-one-vote fitting is not.
                point += normal * (5.0 + 0.10 * (j % 40))
            pts.append(point)
            px.append([320.0 + 80.0 * math.cos(theta), 320.0 + 80.0 * math.sin(theta)])
    return np.asarray(pts), np.rint(np.asarray(px)).astype(np.int32)


def main() -> int:
    config = _config()
    floor = _box_floor_normal(config)

    flat_points, flat_pixels = _annulus(floor, seed=1, dense_bad_sector=True)
    flat_plane, flat_diag = _m3928_select_annulus_plane(
        flat_points, flat_pixels, (320.0, 320.0), config
    )
    assert flat_plane is not None
    assert flat_diag["selected_hypothesis"] == "floor_parallel", flat_diag
    flat_error = _vector_angle_deg(flat_plane.normal, floor)
    assert flat_error < 0.2, flat_error
    expected_offset = -float(np.dot(floor, np.asarray([0.0, 0.0, 720.0])))
    raw_point_weighted_offset = float(np.median(-(flat_points @ floor)))
    assert abs(raw_point_weighted_offset - expected_offset) > 2.0
    assert abs(float(flat_plane.offset) - expected_offset) < 0.8

    axis, _ = _basis(floor)
    tilted_normal = _rotation_about_axis(axis, 20.0) @ floor
    tilted_points, tilted_pixels = _annulus(tilted_normal, seed=2)
    tilted_plane, tilted_diag = _m3928_select_annulus_plane(
        tilted_points, tilted_pixels, (320.0, 320.0), config
    )
    assert tilted_plane is not None
    assert tilted_diag["selected_hypothesis"] == "measured", tilted_diag
    tilted_error = _vector_angle_deg(tilted_plane.normal, tilted_normal)
    assert tilted_error < 1.0, tilted_error

    clean_points, clean_pixels = _annulus(floor, seed=3)
    module = "production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry"
    with patch(
        module + "._m3928_sector_balanced_measured_plane",
        return_value=(None, {"status": "synthetic_measured_failure"}),
    ):
        rescue_plane, rescue_diag = _m3928_select_annulus_plane(
            clean_points, clean_pixels, (320.0, 320.0), config
        )
    assert rescue_plane is not None
    assert rescue_diag["selected_hypothesis"] == "floor_parallel", rescue_diag
    assert rescue_diag["floor_parallel_rescue_applied"] is True, rescue_diag

    print("M39.2.8 synthetic annulus-plane regression: PASS")
    print(f"flat dense-crescent -> {flat_diag['selected_hypothesis']}, normal error={flat_error:.3f} deg")
    print(f"true 20-deg tilt    -> {tilted_diag['selected_hypothesis']}, normal error={tilted_error:.3f} deg")
    print(f"measured failure    -> {rescue_diag['selected_hypothesis']}, rescue={rescue_diag['floor_parallel_rescue_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
