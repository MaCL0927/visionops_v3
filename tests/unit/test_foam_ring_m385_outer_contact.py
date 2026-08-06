from __future__ import annotations

from typing import Any

import numpy as np  # type: ignore

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_surface_outer_contact_m385 import (
    fit_side_surface_outer_contact_m385,
)


def _instance(instance_id: int, class_name: str, mask: np.ndarray) -> SegmentationInstance:
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=0 if class_name == "foam_ring" else 1,
        class_name=class_name,
        confidence=0.96,
        mask=mask.astype(bool),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _hybrid_raw() -> dict[str, Any]:
    return {
        "depth": {"minimum_mm": 150, "maximum_mm": 3000},
        "object_geometry": {
            "nominal_outer_diameter_mm": 85.0,
            "nominal_inner_diameter_mm": 60.0,
            "axial_length_mm": 70.0,
        },
        "m38_branch_a": {"enabled": False, "fallback_to_m36": False},
        "m38_branch_b": {
            "enabled": True,
            "fallback_to_m36": False,
            "maximum_candidates_per_trigger": 2,
            "full_geometry_minimum_observed_opening_coverage": 0.15,
            "full_geometry_side_on_angle_deg": 80.0,
            "full_geometry_side_on_minimum_observed_opening_coverage": 0.25,
        },
        "m38_branch_d": {
            "enabled": True,
            "only_without_mouth": True,
            "require_no_mouth_instances_in_scene": True,
            "stop_after_first_eligible": True,
            "maximum_candidates_per_trigger": 3,
        },
        "m38_branch_c": {
            "enabled": True,
            "fast_terminate": True,
            "fallback_to_m36": False,
            "fallback_to_m37_6": False,
            "operator_action": "turn_or_agitate_box",
        },
        "hybrid_grasp": {
            "enabled": True,
            "prefer_mouth_visible": True,
            "legacy_m36_enabled": False,
            "side_ring_fallback_enabled": False,
            "side_ring_only_unmatched": True,
            "depth_layering": {
                "enabled": True,
                "layer_tolerance_mm": 30.0,
                "mask_erode_px": 0,
                "sample_stride": 1,
                "minimum_valid_points": 5,
                "minimum_valid_ratio": 0.01,
            },
            "bounded_refinement": {"maximum_accurate_refinements_per_trigger": 0},
        },
        "side_ring_template": {},
    }


def test_m385_recovers_outer_contact_from_synthetic_side_cylinder() -> None:
    height, width = 160, 220
    fx = fy = 500.0
    cx, cy = width / 2.0, height / 2.0
    radius_mm = 42.5
    center_z_mm = 650.0
    axial_length_mm = 70.0

    yy, xx = np.indices((height, width))
    ray_x = (xx - cx) / fx
    ray_y = (yy - cy) / fy
    # Camera ray is t * [ray_x, ray_y, 1]. The cylinder axis is camera X.
    a = ray_y * ray_y + 1.0
    b = -2.0 * center_z_mm
    c = center_z_mm * center_z_mm - radius_mm * radius_mm
    discriminant = b * b - 4.0 * a * c
    valid = discriminant >= 0.0
    depth = np.zeros((height, width), dtype=np.uint16)
    t = np.zeros((height, width), dtype=np.float64)
    t[valid] = (-b - np.sqrt(discriminant[valid])) / (2.0 * a[valid])
    valid &= np.abs(t * ray_x) <= 0.5 * axial_length_mm
    depth[valid] = np.rint(t[valid]).astype(np.uint16)
    ring = _instance(0, "foam_ring", valid)

    raw = _hybrid_raw()
    raw["m38_branch_d"].update({
        "ring_mask_erode_px": 1,
        "neighbor_exclusion_dilate_px": 0,
        "side_depth_front_tolerance_mm": 30.0,
        "side_depth_back_tolerance_mm": 80.0,
        "depth_edge_threshold_mm": 20.0,
        "depth_edge_dilate_px": 0,
        "surface_component_minimum_ratio": 0.05,
        "normal_neighbor_step_px": 1,
        "minimum_side_points": 50,
        "minimum_normal_points": 30,
        "maximum_fit_points": 500,
        "maximum_normal_fit_points": 400,
        "central_axis_fraction": 0.75,
        "local_axis_offsets_deg": [-8, -4, 0, 4, 8],
        "fixed_radius_iterations": 4,
        "radial_inlier_threshold_mm": 5.0,
        "minimum_radial_inlier_ratio": 0.70,
        "maximum_radial_residual_median_mm": 3.0,
        "maximum_radial_residual_p90_mm": 8.0,
        "maximum_normal_axis_median_deg": 20.0,
        "maximum_normal_axis_p90_deg": 45.0,
        "maximum_normal_radial_median_deg": 25.0,
        "minimum_visible_normal_span_deg": 40.0,
        "minimum_observed_axis_span_mm": 20.0,
        "minimum_axis_view_angle_deg": 70.0,
        "maximum_normal_axis_eigenvalue_ratio": 0.60,
        "bootstrap_group_count": 4,
        "maximum_bootstrap_axis_dispersion_deg": 15.0,
        "contact_support_axial_half_span_mm": 10.0,
        "maximum_contact_support_error_mm": 8.0,
        "minimum_contact_inside_mask_px": 1.0,
    })
    result = fit_side_surface_outer_contact_m385(
        ring,
        [ring],
        depth,
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        raw,
    )

    assert result["eligible"] is True
    candidate = result["candidate"]
    assert candidate["robot_ready"] is False
    outer = candidate["outer_contact"]
    axis = np.asarray(outer["cylinder_axis_camera_undirected"], dtype=np.float64)
    normal = np.asarray(outer["outer_surface_normal_camera"], dtype=np.float64)
    closing = np.asarray(outer["closing_direction_camera"], dtype=np.float64)
    assert abs(float(np.dot(axis, [1.0, 0.0, 0.0]))) > 0.98
    assert float(normal[2]) < -0.98
    assert np.allclose(closing, -normal, atol=1e-6)
    assert result["diagnostics"]["radial_inlier_ratio"] > 0.95


def test_m385_weak_depth_opening_skips_expensive_geometry_and_returns_outer_contact() -> None:
    mask = np.zeros((60, 100), dtype=bool)
    mask[10:50, 20:72] = True
    ring = _instance(7, "foam_ring", mask)
    inferred_mask = np.zeros_like(mask)
    inferred_mask[25:38, 22:38] = True
    inferred_mouth = _instance(-100007, "ring_mouth", inferred_mask)
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[mask] = 600
    analyze_calls: list[str] = []
    outer_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [], [ring], [], []

    def infer(_ring, _depth, _raw):
        return {
            "eligible": True,
            "mouth_instance": inferred_mouth,
            "association": {"association_mode": "depth_inferred_partial_opening"},
            "rejection_reasons": [],
            "diagnostics": {"opening_source": "depth_inferred", "evidence_score": 4.0},
            "timing_ms": {"total_ms": 1.0},
            "_debug": {},
        }

    def fit(*_args, **_kwargs):
        return {
            "ring_instance_id": 7,
            "mouth_instance_id": -100007,
            "eligible": True,
            "rejection_reasons": [],
            "warnings": [],
            "association": {},
            "diagnostics": {
                "observed_opening_coverage": 0.05,
                "axis_view_angle_deg": 85.0,
            },
            "pose_payload": {"ring_instance_id": 7},
            "synthetic_mouth_instance": inferred_mouth,
            "synthetic_ring_instance": ring,
            "timing_ms": {"total_ms": 2.0},
        }

    def analyze(*_args, **_kwargs):
        analyze_calls.append("called")
        raise AssertionError("weak depth opening must not enter 12-clock geometry")

    def outer_fit(instance, *_args, **_kwargs):
        outer_calls.append(int(instance.instance_id))
        return {
            "ring_instance_id": 7,
            "eligible": True,
            "rejection_reasons": [],
            "warnings": [],
            "diagnostics": {
                "radial_inlier_ratio": 0.9,
                "radial_residual_median_mm": 1.0,
            },
            "candidate": {
                "candidate_type": "outer_contact_geometry",
                "grasp_branch": "m38_5_side_surface_outer_contact_only",
                "grasp_mode": "outer_contact_only",
                "pose_source": "observed_outer_cylinder_side_surface",
                "robot_ready": False,
                "target": {"ring_instance_id": 7},
                "outer_contact": {
                    "contact_camera_mm": [0.0, 0.0, 600.0],
                    "contact_uv": [50.0, 30.0],
                    "outer_surface_normal_camera": [0.0, 0.0, -1.0],
                    "closing_direction_camera": [0.0, 0.0, 1.0],
                    "cylinder_axis_camera_undirected": [1.0, 0.0, 0.0],
                },
                "quality": {"radial_inlier_ratio": 0.9},
            },
            "timing_ms": {"total_ms": 3.0},
        }

    raw = _hybrid_raw()
    scene = run_hybrid_grasp(
        [ring],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        partial_infer_fn=infer,
        partial_fit_fn=fit,
        analyze_fn=analyze,
        outer_contact_fit_fn=outer_fit,
    )

    assert analyze_calls == []
    assert outer_calls == [7]
    assert scene["selected_grasp_branch"] == "m38_5_side_surface_outer_contact_only"
    assert scene["m38_3_branch_b"]["fast_gate_skipped_ring_ids"] == [7]
    assert scene["m38_3_branch_b"]["timing_ms"]["rim_pinch_geometry_ms"] == 0.0
    assert scene["m38_5_branch_d"]["attempt_count"] == 1
    assert scene["robot_candidate"]["robot_ready"] is False


def test_m385_does_not_take_over_scene_when_ring_mouth_exists() -> None:
    ring_mask = np.zeros((60, 100), dtype=bool)
    ring_mask[10:50, 20:72] = True
    mouth_mask = np.zeros_like(ring_mask)
    mouth_mask[22:38, 38:54] = True
    ring = _instance(7, "foam_ring", ring_mask)
    mouth = _instance(8, "ring_mouth", mouth_mask)
    depth = np.zeros(ring_mask.shape, dtype=np.uint16)
    depth[ring_mask] = 600
    outer_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [], [ring], [mouth], []

    def infer(*_args, **_kwargs):
        return {
            "eligible": False,
            "mouth_instance": None,
            "association": {},
            "rejection_reasons": ["none"],
            "diagnostics": {},
            "timing_ms": {"total_ms": 1.0},
            "_debug": {},
        }

    def outer_fit(instance, *_args, **_kwargs):
        outer_calls.append(int(instance.instance_id))
        raise AssertionError("M38.5 is scene-gated when any mouth instance exists")

    raw = _hybrid_raw()
    scene = run_hybrid_grasp(
        [ring, mouth],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        partial_infer_fn=infer,
        outer_contact_fit_fn=outer_fit,
    )

    assert outer_calls == []
    assert scene["m38_5_branch_d"]["scene_gate_passed"] is False
    assert scene["selected_grasp_branch"] == "m38_4_branch_c_fast_reject"
