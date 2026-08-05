from __future__ import annotations

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_template import (
    SideRingTemplateConfig,
    _axis_angle_deg,
    _fit_axis_hollow_multisurface,
)


def _config() -> SideRingTemplateConfig:
    return SideRingTemplateConfig.from_mapping(
        {
            "object_geometry": {
                "nominal_outer_diameter_mm": 85.0,
                "nominal_inner_diameter_mm": 60.0,
                "axial_length_mm": 70.0,
            },
            "side_ring_template": {
                "multi_surface_enabled": True,
                "minimum_side_lay_angle_deg": 0.0,
                "screen_axis_hypothesis_count": 12,
                "screen_local_refine_angles_deg": [5.0, 1.5],
            },
        }
    )


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(axis[2])) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u); v /= np.linalg.norm(v)
    return u, v


def test_m376_joint_fit_recovers_axis_from_outer_inner_and_faces():
    rng = np.random.default_rng(376)
    axis = np.array([-0.55, 0.34, 0.76], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    u, v = _basis(axis)
    center = np.array([25.0, 20.0, 650.0])
    outer_r, inner_r, half = 42.5, 30.0, 35.0

    points = []
    normals = []
    normal_points = []
    for radius, inward in ((outer_r, False), (inner_r, True)):
        for _ in range(500):
            t = rng.uniform(-half, half)
            angle = rng.uniform(-1.2, 1.5)
            radial = np.cos(angle) * u + np.sin(angle) * v
            point = center + t * axis + radius * radial + rng.normal(0, 0.6, 3)
            normal = (-radial if inward else radial) + rng.normal(0, 0.02, 3)
            normal /= np.linalg.norm(normal)
            points.append(point); normal_points.append(point); normals.append(normal)
    for sign in (-1.0, 1.0):
        for _ in range(260):
            radius = np.sqrt(rng.uniform(inner_r**2, outer_r**2))
            angle = rng.uniform(-1.1, 1.3)
            radial = np.cos(angle) * u + np.sin(angle) * v
            point = center + sign * half * axis + radius * radial + rng.normal(0, 0.5, 3)
            normal = sign * axis + rng.normal(0, 0.02, 3)
            normal /= np.linalg.norm(normal)
            points.append(point); normal_points.append(point); normals.append(normal)

    config = _config()
    evaluation, timing = _fit_axis_hollow_multisurface(
        np.asarray(points),
        np.asarray(normal_points),
        np.asarray(normals),
        config,
        profile="screen",
        initial_axis=None,
        mouth_support=None,
    )
    assert _axis_angle_deg(evaluation.axis, axis) < 5.0
    assert evaluation.surface_inlier_ratio > 0.75
    assert evaluation.surface_counts["outer"] > 200
    assert evaluation.surface_counts["inner"] > 150
    assert evaluation.surface_counts["near_face"] > 50
    assert evaluation.surface_counts["far_face"] > 50
    assert timing["fit_model"] == "hollow_short_cylinder_multisurface"
