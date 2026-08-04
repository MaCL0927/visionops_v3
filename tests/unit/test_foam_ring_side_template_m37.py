import math

import cv2
import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (
    SegmentationInstance,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_template import (
    SideRingTemplateConfig,
    fit_side_ring_instance,
    select_best_side_ring,
)


def _synthetic_side_cylinder():
    height = width = 320
    intrinsics = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 160.0}
    center = np.asarray([30.0, 0.0, 600.0])
    axis = np.asarray([1.0, 0.0, 0.0])
    radius = 42.5
    length = 70.0
    points = []
    for axial in np.linspace(-length / 2.0, length / 2.0, 120):
        for angle in np.linspace(0.0, 2.0 * math.pi, 240, endpoint=False):
            # Full surface is rasterized; the z-buffer retains the visible half.
            points.append(
                center
                + axis * axial
                + np.asarray([0.0, math.sin(angle), math.cos(angle)]) * radius
            )
    points = np.asarray(points)
    uv = np.column_stack(
        (
            intrinsics["fx"] * points[:, 0] / points[:, 2] + intrinsics["cx"],
            intrinsics["fy"] * points[:, 1] / points[:, 2] + intrinsics["cy"],
        )
    )
    depth = np.zeros((height, width), dtype=np.uint16)
    for point, pixel in zip(points, uv):
        u, v = np.rint(pixel).astype(int)
        if 0 <= u < width and 0 <= v < height:
            z = int(round(point[2]))
            if depth[v, u] == 0 or z < int(depth[v, u]):
                depth[v, u] = z
    mask = depth > 0
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    ys, xs = np.nonzero(mask)
    instance = SegmentationInstance(
        instance_id=3,
        class_id=0,
        class_name="foam_ring",
        confidence=0.99,
        mask=mask,
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )
    raw = {
        "object_geometry": {
            "nominal_outer_diameter_mm": 85.0,
            "nominal_inner_diameter_mm": 60.0,
            "axial_length_mm": 70.0,
        },
        "side_ring_template": {
            "mask_erode_px": 0,
            "global_axis_samples": 160,
            "local_refine_angles_deg": [8.0, 2.5],
            "maximum_fit_points": 1000,
            "minimum_radial_inlier_ratio": 0.40,
            "maximum_radial_residual_median_mm": 6.0,
            "maximum_radial_residual_p90_mm": 20.0,
            "minimum_observed_axis_span_mm": 25.0,
            "minimum_side_lay_angle_deg": 45.0,
        },
    }
    return instance, depth, intrinsics, SideRingTemplateConfig.from_mapping(raw)


def test_m37_fits_side_axis_and_directs_it_to_nearer_endpoint():
    instance, depth, intrinsics, config = _synthetic_side_cylinder()
    result = fit_side_ring_instance(instance, depth, intrinsics, config)
    axis = np.asarray(result["axis_toward_camera"])
    assert result["eligible"] is True
    assert abs(float(axis[0])) > 0.92
    # The cylinder center is at +X. Its -X endpoint is closer to the camera origin.
    assert float(axis[0]) < 0.0
    assert result["near_endpoint_camera_distance_mm"] < result["far_endpoint_camera_distance_mm"]
    assert result["axis_view_angle_deg"] > 80.0
    crown = result["near_side_crown"]
    axis = np.asarray(result["axis_toward_camera"], dtype=np.float64)
    near = np.asarray(result["near_opening_center_camera_mm"], dtype=np.float64)
    grasp = np.asarray(crown["grasp_point_camera_mm"], dtype=np.float64)
    axial_inset = float(np.dot(near - grasp, axis))
    axis_point = near - axis * axial_inset
    radial_distance = float(np.linalg.norm(grasp - axis_point))
    assert abs(axial_inset - config.grasp_axial_inset_mm) < 1e-5
    assert abs(radial_distance - config.outer_radius_mm) < 1e-5
    assert crown["radius_mode"] == "outer_surface"
    assert crown["direction_source"] in {
        "visible_surface_angular_interval",
        "camera_facing_fallback",
    }
    # The old near-opening projected top point remains available for diagnosis.
    assert result["near_opening_rim_top_diagnostic"]["point_uv"][1] < result["near_opening_center_uv"][1]
    # The compatibility alias must point to the corrected M37.1 grasp point.
    assert result["top_arc"]["grasp_point_uv"] == crown["grasp_point_uv"]


def test_m37_marks_mouth_matched_ring_for_m36_preference():
    instance, depth, intrinsics, config = _synthetic_side_cylinder()
    result = fit_side_ring_instance(
        instance,
        depth,
        intrinsics,
        config,
        mouth_matched=True,
    )
    assert result["eligible"] is False
    assert "mouth_matched_prefer_m36_branch" in result["rejection_reasons"]


def test_m37_selects_nearest_eligible_side_ring():
    selected = select_best_side_ring(
        [
            {"ring_instance_id": 1, "eligible": True, "near_endpoint_camera_distance_mm": 650.0, "fit_score": 1.0},
            {"ring_instance_id": 2, "eligible": False, "near_endpoint_camera_distance_mm": 500.0, "fit_score": 0.5},
            {"ring_instance_id": 3, "eligible": True, "near_endpoint_camera_distance_mm": 610.0, "fit_score": 3.0},
        ]
    )
    assert selected is not None
    assert selected["ring_instance_id"] == 3
