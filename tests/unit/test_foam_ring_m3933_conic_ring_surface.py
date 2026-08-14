import math
import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.conic_ring_surface import (
    reconstruct_conic_ring_surface,
)


def _scene():
    h = w = 200
    cx = cy = 100.0
    fx = fy = 600.0
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    yy, xx = np.indices((h, w))
    rr = np.hypot(xx - cx, yy - cy)
    mouth = rr <= 24
    # Semantic silhouette intentionally includes a long visible side wall.
    ring = (rr >= 24) & (rr <= 58)
    obj = {
        "nominal_inner_diameter_mm": 60.0,
        "nominal_outer_diameter_mm": 85.0,
        "nominal_wall_thickness_mm": 14.0,
        "axial_length_mm": 70.0,
    }
    cfg = {
        "sector_count": 16,
        "minimum_surface_sectors": 5,
        "classification_minimum_sectors": 6,
        "classification_minimum_coverage_deg": 135.0,
        "classification_maximum_residual_mm": 5.0,
        "classification_maximum_jackknife_tilt_std_deg": 8.0,
        "minimum_predicted_annulus_semantic_support_ratio": 0.20,
        "classification_minimum_semantic_support_ratio": 0.20,
        "classification_maximum_conic_radius_error_ratio": 0.35,
        "classification_maximum_circle_residual_p90_mm": 7.0,
        "classification_maximum_reprojection_p90_px": 6.0,
        "classification_minimum_annulus_depth_support_sectors": 4,
        "classification_minimum_annulus_depth_support_ratio": 0.20,
        "classification_maximum_annulus_depth_residual_median_mm": 7.0,
        "flat_minimum_annulus_depth_support_sectors": 4,
        "flat_minimum_annulus_depth_support_ratio": 0.20,
        "flat_maximum_annulus_depth_residual_median_mm": 7.0,
        "flat_maximum_conic_radius_error_ratio": 0.20,
        "require_m39_3_1_tilt_consensus": True,
    }
    return (cx, cy), intr, xx, yy, rr, mouth, ring, obj, cfg


def _depth(tilt_deg, xx, yy, rr, ring, cx=100.0, cy=100.0, fx=600.0):
    gx = math.tan(math.radians(tilt_deg))
    z = 750.0 / (1.0 - gx * (xx - cx) / fx)
    depth = np.zeros(rr.shape, np.uint16)
    front = (rr >= 24) & (rr <= 34)
    depth[front] = np.rint(z[front]).astype(np.uint16)
    # Global foam_ring silhouette extends far beyond the actual front face.
    side = (rr > 34) & ring
    depth[side] = np.rint(z[side] + 70.0).astype(np.uint16)
    # Four sectors expose only the lower/far surface. A correct reconstruction
    # must leave those unsupported instead of forming a giant false plane.
    theta = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    sector = np.floor(theta * 16 / (2 * np.pi)).astype(int)
    corrupt = front & np.isin(sector, [0, 1, 8, 9])
    depth[corrupt] = np.rint(z[corrupt] + 70.0).astype(np.uint16)
    return depth


def _run(tilt_deg, prior_state):
    center, intr, xx, yy, rr, mouth, ring, obj, cfg = _scene()
    axes = np.eye(3)
    return reconstruct_conic_ring_surface(
        _depth(tilt_deg, xx, yy, rr, ring),
        ring,
        mouth,
        center,
        intr,
        box_x_camera=axes[0],
        box_y_camera=axes[1],
        box_z_inside_camera=axes[2],
        object_geometry=obj,
        config=cfg,
        prior_tilt_evidence={"state": prior_state, "confidence": "strong"},
    )


def test_m3933_ignores_global_sidewall_silhouette_and_far_layer():
    result = _run(15.0, "TILTED")
    assert result["production_routing_enabled"] is False
    assert result["foam_ring_outer_silhouette_used_as_front_outer_rim"] is False
    assert result["classification"] == "TILTED"
    surface = result["surface"]
    assert 9.0 <= float(surface["tilt_deg"]) <= 22.0
    assert float(surface["conic_validation"]["inner_radius_error_ratio"]) < 0.20
    selected = [
        row["selected_candidate"]
        for row in surface["sector_samples"]
        if row.get("selected_candidate")
    ]
    assert selected
    assert max(float(row["depth_mm"]) for row in selected) < 800.0


def test_m3933_independent_ring_gradient_vetoes_ambiguous_tilt():
    result = _run(15.0, "FLAT")
    assert 9.0 <= float(result["surface"]["tilt_deg"]) <= 22.0
    assert result["classification"] == "UNCERTAIN"
    assert result["reason"] == "conic_tilt_conflicts_with_independent_ring_gradient"
    assert result["independent_tilt_crosscheck"]["tilt_consensus_ok"] is False


def test_m3933_flat_requires_target_annulus_depth_support():
    result = _run(0.0, "FLAT")
    assert result["classification"] == "FLAT"
    assert float(result["surface"]["tilt_deg"]) < 3.0
    depth_support = result["surface"]["conic_validation"]["predicted_annulus_depth_support"]
    assert int(depth_support["supported_sector_count"]) >= 4
