#!/usr/bin/env python3
"""Synthetic smoke tests for the M39.2.9 Branch-A front-surface model."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    _box_reference_axes_camera,
    _m3929_floor_constrained_front_surface,
    _m3929_select_front_depth_layer,
    _vector_angle_deg,
)

BOX = REPO_ROOT / "production/foam_ring_grasp/config/box_model.json"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0])
    u = _unit(helper - normal * float(np.dot(helper, normal)))
    v = _unit(np.cross(normal, u))
    return u, v


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _unit(axis)
    theta = math.radians(angle_deg)
    kx, ky, kz = axis
    k = np.asarray([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return _unit((np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)) @ vector)


def _ring(normal: np.ndarray, center: np.ndarray, seed: int, biases: dict[int, float] | None = None):
    rng = np.random.default_rng(seed)
    u, v = _basis(normal)
    points, pixels = [], []
    for sector in range(16):
        for j in range(12):
            theta = 2.0 * math.pi * (sector + (j + 0.5) / 12.0) / 16.0
            p = center + 46.0 * (math.cos(theta) * u + math.sin(theta) * v)
            p += normal * (float((biases or {}).get(sector, 0.0)) + rng.normal(0.0, 0.5))
            points.append(p)
            pixels.append([320.0 + 80.0 * math.cos(theta), 320.0 + 80.0 * math.sin(theta)])
    return np.asarray(points), np.rint(np.asarray(pixels)).astype(np.int32)


def main() -> int:
    box = json.loads(BOX.read_text(encoding="utf-8"))
    cfg = GeometryConfig({
        "plane": {"ransac_iterations": 320, "inlier_threshold_mm": 5.0, "random_seed": 3401},
        "box_wall": {"model": "calibrated_3d_cuboid", "calibrated_model": box},
        "branch_a_front_surface": {
            "enabled": True,
            "mode": "floor_constrained",
            "depth_layer": {
                "minimum_center_gap_mm": 24.0,
                "minimum_layer_fraction": 0.08,
                "minimum_near_points": 40,
            },
            "height_estimation": {
                "sector_count": 16,
                "minimum_sector_points": 5,
                "minimum_good_sectors": 4,
                "minimum_height_outlier_threshold_mm": 5.0,
                "maximum_height_outlier_threshold_mm": 20.0,
                "mad_scale": 3.0,
                "point_fallback_enabled": True,
                "minimum_point_fallback_count": 40,
                "plane_inlier_threshold_mm": 8.0,
            },
            "measured_plane_diagnostic": {
                "enabled": True,
                "warning_disagreement_deg": 10.0,
                "severe_disagreement_deg": 20.0,
            },
        },
    })
    x_right, _y, z_inside = _box_reference_axes_camera(cfg)
    floor = -z_inside
    center = np.asarray([-50.0, 20.0, 755.0])

    front, px = _ring(floor, center, 1)
    far = front + 67.0 * z_inside
    selected, selected_px, layer = _m3929_select_front_depth_layer(
        np.vstack([front, far]), np.vstack([px, px]), cfg
    )
    assert layer["applied"] and layer["center_gap_mm"] > 50.0
    assert len(selected) == len(selected_px) == len(front)

    tilted = _rotate(floor, x_right, 25.0)
    tilted_points, tilted_px = _ring(tilted, center, 2)
    plane, diag = _m3929_floor_constrained_front_surface(tilted_points, tilted_px, (320.0, 320.0), cfg)
    assert plane is not None
    assert _vector_angle_deg(plane.normal, floor) < 1e-6
    assert diag["measured_plane_diagnostic"]["measured_vs_floor_deg"] > 20.0

    polluted, polluted_px = _ring(floor, center, 3, {0: -45.0, 1: -38.0, 8: 35.0})
    plane2, diag2 = _m3929_floor_constrained_front_surface(polluted, polluted_px, (320.0, 320.0), cfg)
    assert plane2 is not None
    assert abs(float(plane2.offset) + float(np.dot(floor, center))) < 2.0
    assert len(diag2["rejected_sectors"]) >= 2

    print("M39.2.9 synthetic front-surface regression: PASS")
    print(f"two-layer gap      : {layer['center_gap_mm']:.2f} mm -> near layer selected")
    print(f"bad measured tilt  : {diag['measured_plane_diagnostic']['measured_vs_floor_deg']:.2f} deg -> final floor normal")
    print(f"polluted sectors   : rejected={diag2['rejected_sectors']} -> height preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
