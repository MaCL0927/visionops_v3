import math

import cv2
import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.analytic_conic_surface import (
    reconstruct_analytic_conic_surface,
)


def _config():
    return {
        "depth_anchor_inner_guard_px": 2,
        "depth_anchor_wall_fraction": 0.70,
        "depth_anchor_minimum_pixels": 24,
        "depth_anchor_sector_count": 16,
        "depth_anchor_cluster_gap_mm": 3.0,
        "depth_anchor_cluster_minimum_pixels": 5,
        "depth_anchor_minimum_sectors": 5,
        "dense_annulus_inner_fraction": 0.20,
        "dense_annulus_outer_fraction": 0.75,
        "dense_annulus_polygon_points": 160,
        "dense_depth_inlier_mm": 8.0,
        "dense_depth_sector_count": 16,
        "dense_sector_radial_fractions": [0.30, 0.50, 0.70],
        "dense_sector_patch_radius_px": 1,
        "minimum_mouth_band_sectors": 5,
        "minimum_mouth_band_coverage_deg": 112.5,
        "maximum_circle_residual_p90_mm": 7.0,
        "maximum_reprojection_p90_px": 7.0,
        "maximum_mouth_band_residual_median_mm": 20.0,
        "minimum_dense_inlier_ratio_for_rescue": 0.40,
        "minimum_semantic_support_ratio": 0.30,
        "minimum_valid_depth_ratio": 0.22,
        "minimum_inlier_ratio": 0.28,
        "minimum_angular_coverage_deg": 112.5,
        "maximum_residual_median_mm": 10.0,
        "mouth_band_score_scale_mm": 10.0,
        "strong_flat_minimum_sectors": 8,
        "strong_flat_maximum_mouth_band_residual_mm": 4.5,
        "near_flat_analytic_max_deg": 10.0,
        "definite_analytic_tilt_min_deg": 12.0,
        "minimum_analytic_ab_margin": 0.035,
        "minimum_analytic_ab_margin_with_m3931": 0.020,
        "m3931_supportive_required_margin": 0.025,
        "off_axis_flat_protection_deg": 12.0,
        "maximum_accepted_tilt_deg": 35.0,
    }


def _synthetic_ring(tilt_deg: float, *, direction_deg: float = 0.0, corrupt_far_layer: bool = False):
    h, w = 480, 640
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    a = math.radians(tilt_deg)
    phi = math.radians(direction_deg)
    n_into = np.asarray([
        math.sin(a) * math.cos(phi),
        math.sin(a) * math.sin(phi),
        math.cos(a),
    ])
    n_toward = -n_into
    e1 = np.asarray([1.0, 0.0, 0.0])
    e1 = e1 - n_toward * float(np.dot(e1, n_toward))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_toward, e1)
    e2 /= np.linalg.norm(e2)
    center = np.asarray([0.0, 0.0, 750.0])
    angles = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=False)

    def projected(radius):
        pts = center[None, :] + radius * (
            np.cos(angles)[:, None] * e1[None, :] + np.sin(angles)[:, None] * e2[None, :]
        )
        return np.column_stack([fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy])

    inner = projected(30.0)
    outer = projected(42.5)
    mouth = np.zeros((h, w), np.uint8)
    ring = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mouth, [np.rint(inner).astype(np.int32)], 1)
    cv2.fillPoly(ring, [np.rint(outer).astype(np.int32)], 1)
    cv2.fillPoly(ring, [np.rint(inner).astype(np.int32)], 0)

    yy, xx = np.indices((h, w))
    denom = n_toward[0] * (xx - cx) / fx + n_toward[1] * (yy - cy) / fy + n_toward[2]
    d = float(np.dot(n_toward, center))
    z = d / denom
    depth = np.zeros((h, w), np.uint16)
    front = ring.astype(bool)
    depth[front] = np.rint(z[front]).astype(np.uint16)

    # Simulate lower-ring / side-wall contamination in four angular sectors.
    # M39.3.4 must not turn that +70 mm layer into a giant false plane.
    if corrupt_far_layer:
        theta = (np.arctan2(yy - cy, xx - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        sector = np.floor(theta * 16 / (2.0 * np.pi)).astype(int)
        bad = front & np.isin(sector, [0, 1, 8, 9])
        depth[bad] = np.rint(z[bad] + 70.0).astype(np.uint16)

    return depth, ring.astype(bool), mouth.astype(bool), intr


def _run(tilt_deg, prior_state="UNCERTAIN", corrupt_far_layer=False):
    depth, ring, mouth, intr = _synthetic_ring(tilt_deg, corrupt_far_layer=corrupt_far_layer)
    return reconstruct_analytic_conic_surface(
        depth,
        ring,
        mouth,
        (320.0, 240.0),
        intr,
        box_x_camera=[1.0, 0.0, 0.0],
        box_y_camera=[0.0, 1.0, 0.0],
        box_z_inside_camera=[0.0, 0.0, 1.0],
        object_geometry={"nominal_inner_diameter_mm": 60.0, "nominal_outer_diameter_mm": 85.0},
        config=_config(),
        prior_tilt_evidence={"state": prior_state, "confidence": "strong"},
    )


def test_m3934_flat_reference_is_resolved_without_zero_value_bug():
    result = _run(0.0, "FLAT")
    assert result["production_routing_enabled"] is False
    assert result["classification"] == "FLAT"
    assert float(result["analytic_min_tilt_deg"]) < 1.0
    assert result["selected_candidate"]["candidate_label"] == "FLAT_REFERENCE"


def test_m3934_dense_depth_resolves_analytic_ab_tilt():
    result = _run(15.0, "UNCERTAIN")
    assert result["classification"] == "TILTED"
    assert result["selected_candidate"]["candidate_label"] in {"CONIC_A", "CONIC_B"}
    assert 11.0 <= float(result["selected_candidate"]["tilt_deg"]) <= 20.0
    assert float(result["winner_margin"]) > 0.035


def test_m3934_m3931_is_auxiliary_not_hard_veto():
    result = _run(15.0, "FLAT")
    assert result["prior_m39_3_1"]["hard_veto"] is False
    assert result["classification"] == "TILTED"


def test_m3934_partial_far_layer_does_not_create_giant_tilt():
    result = _run(15.0, "TILTED", corrupt_far_layer=True)
    assert result["classification"] == "TILTED"
    assert float(result["selected_candidate"]["tilt_deg"]) < 25.0
    assert int(result["candidate_count"]) <= 3
