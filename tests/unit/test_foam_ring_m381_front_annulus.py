from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (
    GeometryConfig,
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (
    HybridGraspConfig,
    run_hybrid_grasp,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (
    SegmentationInstance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LINE_CONFIG = REPO_ROOT / "production/foam_ring_grasp/config/line.yaml"


def _instance(
    instance_id: int,
    class_id: int,
    class_name: str,
    mask: np.ndarray,
) -> SegmentationInstance:
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.95,
        mask=mask.astype(bool),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _geometry_config() -> GeometryConfig:
    return GeometryConfig(
        {
            "classes": {"foam_ring": "foam_ring", "ring_mouth": "ring_mouth"},
            "association": {
                "use_filled_outer_envelope": True,
                "envelope_close_px": 2,
                "envelope_dilate_px": 2,
                "minimum_component_area_ratio": 0.03,
                "minimum_envelope_containment": 0.55,
                "minimum_center_inside": True,
                "maximum_normalized_center_distance": 0.60,
                "minimum_mouth_to_ring_area_ratio": 0.02,
                "maximum_mouth_to_ring_area_ratio": 0.70,
            },
            "depth": {
                "minimum_mm": 150,
                "maximum_mm": 3000,
                "minimum_valid_points": 40,
                "local_obstacle_margin_mm": 8.0,
                "maximum_front_obstacle_ratio": 0.20,
                "local_obstacle_observable_max_tilt_deg": 30.0,
            },
            "plane": {
                "ransac_iterations": 120,
                "inlier_threshold_mm": 2.0,
                "minimum_inlier_ratio": 0.45,
                "random_seed": 7,
            },
            "pose": {"normal_mode": "depth_plane"},
            "m38_branch_a": {
                "enabled": True,
                "fallback_to_m36": True,
                "require_strict_association": True,
                "minimum_mouth_containment": 0.65,
                "minimum_mouth_to_ring_area_ratio": 0.04,
                "maximum_mouth_to_ring_area_ratio": 0.58,
                "minimum_mouth_minor_major_ratio": 0.35,
                "maximum_ellipse_normalized_residual_p90": 0.38,
                "ring_mask_erode_px": 1,
                "annulus_outer_expand_ratio": 0.55,
                "minimum_annulus_outer_expand_px": 7,
                "maximum_annulus_outer_expand_px": 32,
                "mouth_inner_exclusion_px": 1,
                "exclude_other_ring_masks": True,
                "neighbor_exclusion_dilate_px": 1,
                "depth_edge_threshold_mm": 30.0,
                "depth_edge_dilate_px": 0,
                "point_sample_stride": 1,
                "front_depth_layer_separation_enabled": True,
                "front_depth_layer_minimum_gap_mm": 18.0,
                "front_depth_layer_minimum_fraction": 0.15,
                "front_depth_layer_minimum_near_points": 40,
                "minimum_front_layer_angular_coverage_deg": 135.0,
                "minimum_front_layer_inlier_angular_coverage_deg": 135.0,
                "angular_sector_count": 16,
                "minimum_angular_coverage_deg": 180.0,
                "minimum_inlier_angular_coverage_deg": 150.0,
                "minimum_valid_points": 40,
                "minimum_depth_valid_ratio": 0.20,
                "minimum_plane_inlier_ratio": 0.40,
                "maximum_plane_residual_p95_mm": 4.0,
            },
            "_runtime": {"pose_strategy": "m38_1_front_annulus"},
            "object_geometry": {
                "minimum_inner_diameter_mm": 30.0,
                "maximum_inner_diameter_mm": 120.0,
                "minimum_wall_thickness_mm": 8.0,
                "maximum_wall_thickness_mm": 55.0,
                "maximum_outer_search_radius_ratio": 1.6,
                "physical_size_hard_reject": False,
            },
            "gripper": {
                "finger_thickness_mm": 17.0,
                "finger_width_mm": 20.0,
                "opening_reference": "inner_gap",
                "minimum_opening_mm": 10.0,
                "maximum_opening_mm": 80.0,
                "opening_safety_margin_mm": 3.0,
                "wall_compression_mm": 1.5,
                "rim_insert_depth_mm": 20.0,
                "pregrasp_offset_mm": 90.0,
                "preopen_clearance_mm": 6.0,
                "clock_position_count": 12,
                "geometry_candidate_max_tilt_deg": 45.0,
                "robot_safe_max_tilt_deg": 30.0,
                "top_layer_tolerance_mm": 15.0,
            },
            "box_wall": {"enabled": False},
            "candidate": {
                "minimum_inner_finger_mouth_containment": 0.55,
                "maximum_other_ring_overlap_ratio": 0.12,
                "minimum_image_border_margin_px": 2,
                "minimum_neighbor_clearance_mm": 0.0,
                "hard_reject_front_obstacle": False,
            },
            "quality": {
                "minimum_depth_valid_ratio": 0.25,
                "minimum_ellipse_points": 12,
                "minimum_mouth_area_px": 80,
                "minimum_ring_area_px": 300,
            },
            "geometry_optimization": {
                "enabled": True,
                "mode": "first_valid",
                "skip_rejected_pairs": True,
                "stop_after_first_valid_target": True,
                "stop_after_first_valid_candidate": True,
                "maximum_pairs_to_fully_analyze": 3,
                "maximum_full_candidates_per_pair": 12,
            },
            "clock_search": {
                "mode": "adaptive_8_plus_4",
                "primary_clock_hours": [12, 2, 3, 5, 6, 8, 9, 11],
                "fallback_to_remaining": True,
            },
            "robot_interface": {"camera_frame_id": "camera_color_optical_frame"},
        }
    )


def _synthetic_scene(*, flat_mouth: bool = False):
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    if flat_mouth:
        cv2.ellipse(mouth, (180, 130), (22, 5), 0.0, 0.0, 360.0, 1, -1)
    else:
        cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    depth[mouth.astype(bool)] = 860
    instances = [
        _instance(0, 0, "foam_ring", ring),
        _instance(1, 1, "ring_mouth", mouth),
    ]
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    return instances, depth, intrinsics


def test_m381_production_configuration_enables_branch_a_with_m384_legacy_disabled() -> None:
    raw = load_yaml(LINE_CONFIG)
    hybrid = HybridGraspConfig.from_mapping(raw)
    assert raw["schema_version"] == "6.5"
    assert hybrid.m38_branch_a_enabled is True
    assert hybrid.m38_branch_a_fallback_to_m36 is False
    assert hybrid.legacy_m36_enabled is False
    assert raw["hybrid_grasp"]["side_ring_fallback_enabled"] is False


def test_m381_clear_opening_uses_direct_3d_annulus_plane() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    scene = analyze_scene(instances, depth, intrinsics, _geometry_config())
    assert scene["pose_strategy"] == "m38_1_front_annulus"
    assert scene["eligible_count"] == 1
    item = scene["instances"][0]
    assert item["pose_strategy"] == "m38_1_front_annulus"
    assert item["m38_branch_a"]["opening_clear"] is True
    assert item["m38_branch_a"]["angular_coverage_deg"] == pytest.approx(360.0)
    assert item["m38_branch_a"]["inlier_angular_coverage_deg"] == pytest.approx(360.0)
    assert item["plane"]["inlier_ratio"] == pytest.approx(1.0)
    normal = np.asarray(item["ring_axis_toward_camera"], dtype=np.float64)
    assert normal == pytest.approx(np.asarray([0.0, 0.0, -1.0]), abs=1e-6)
    assert item["pose"]["normal_source"] == "m38_1_front_annulus_depth_plane"


def test_m3922_strong_two_layer_annulus_keeps_camera_near_front_face() -> None:
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[ring.astype(bool)] = 800
    ys, _xs = np.indices((height, width))
    deeper_inner_side = ring.astype(bool) & (ys >= 130)
    depth[deeper_inner_side] = 860
    depth[mouth.astype(bool)] = 870
    instances = [
        _instance(0, 0, "foam_ring", ring),
        _instance(1, 1, "ring_mouth", mouth),
    ]
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}

    scene = analyze_scene(instances, depth, intrinsics, _geometry_config())
    item = scene["instances"][0]
    layer = item["m38_branch_a"]["front_depth_layer"]
    assert layer["applied"] is True
    assert layer["depth_gap_mm"] == pytest.approx(60.0)
    assert layer["near_fraction"] == pytest.approx(0.5, abs=0.03)
    assert item["plane"]["centroid_camera_mm"][2] == pytest.approx(800.0, abs=1.0)
    assert item["tilt_deg"] == pytest.approx(0.0, abs=0.2)
    assert scene["robot_candidate"] is not None


def test_m3922_robot_safe_tilt_is_hard_gate_for_robot_candidate() -> None:
    height, width = 260, 360
    ring = np.zeros((height, width), dtype=np.uint8)
    mouth = np.zeros_like(ring)
    cv2.circle(ring, (180, 130), 42, 1, -1)
    cv2.circle(mouth, (180, 130), 20, 1, -1)
    intrinsics = {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 130.0}
    theta = np.deg2rad(35.0)
    normal = np.asarray([np.sin(theta), 0.0, -np.cos(theta)], dtype=np.float64)
    center = np.asarray([0.0, 0.0, 800.0], dtype=np.float64)
    offset = -float(np.dot(normal, center))
    depth = np.zeros((height, width), dtype=np.uint16)
    ys, xs = np.nonzero(ring)
    for y, x in zip(ys, xs):
        ray = np.asarray([(x - 180.0) / 400.0, (y - 130.0) / 400.0, 1.0])
        scale = -offset / float(np.dot(normal, ray))
        depth[y, x] = int(round(scale))
    depth[mouth.astype(bool)] = 900
    instances = [
        _instance(0, 0, "foam_ring", ring),
        _instance(1, 1, "ring_mouth", mouth),
    ]

    scene = analyze_scene(instances, depth, intrinsics, _geometry_config())
    item = scene["instances"][0]
    assert item["tilt_deg"] == pytest.approx(35.0, abs=0.5)
    assert item["eligible"] is True
    assert item["robot_safe_tilt"] is False
    assert item["robot_eligible"] is False
    assert "tilt_above_robot_safe_limit" in item["warnings"]
    assert scene["eligible_count"] == 1
    assert scene["robot_eligible_count"] == 0
    assert scene["robot_candidate"] is None


def test_m381_flat_partial_opening_is_rejected_before_grasp_search() -> None:
    instances, depth, intrinsics = _synthetic_scene(flat_mouth=True)
    scene = analyze_scene(instances, depth, intrinsics, _geometry_config())
    item = scene["instances"][0]
    assert item["eligible"] is False
    assert "m38a_mouth_ellipse_too_flat" in item["rejection_reasons"]
    assert item["grasp"]["clock_candidates"] == []


def _hybrid_raw() -> dict[str, Any]:
    return {
        "m38_branch_a": {"enabled": True, "fallback_to_m36": True},
        "hybrid_grasp": {
            "enabled": True,
            "prefer_mouth_visible": True,
            "side_ring_fallback_enabled": True,
            "side_ring_only_unmatched": True,
            "depth_layering": {
                "enabled": True,
                "layer_tolerance_mm": 30.0,
                "mask_erode_px": 0,
                "sample_stride": 1,
                "minimum_valid_points": 5,
                "minimum_valid_ratio": 0.01,
            },
            "bounded_refinement": {
                "maximum_accurate_refinements_per_trigger": 0,
            },
        },
        "side_ring_template": {},
    }


def test_m381_hybrid_prioritizes_branch_a_before_m36_and_m376() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    ring, mouth = instances
    calls: list[str] = []
    side_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {"association_score": 1.0})], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        if strategy == "m38_1_front_annulus":
            candidate = {
                "message_type": "foam_ring_rim_pinch_grasp_candidate",
                "target": {"ring_instance_id": 0, "mouth_instance_id": 1},
            }
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(int(instance.instance_id))
        return {"ring_instance_id": int(instance.instance_id), "eligible": False}

    raw = _hybrid_raw()
    scene = run_hybrid_grasp(
        instances,
        depth,
        intrinsics,
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert calls == ["m38_1_front_annulus"]
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m38_1_clear_mouth_front_annulus_rim_pinch"
    assert scene["robot_candidate"]["pose_source"] == "m38_1_front_annulus_depth_plane"
    assert scene["m38_1_branch_a"]["legacy_m36_retained"] is True
    assert scene["m38_1_branch_a"]["m37_6_retained"] is True


def test_m381_hybrid_falls_back_to_legacy_m36_when_annulus_has_no_candidate() -> None:
    instances, depth, intrinsics = _synthetic_scene()
    ring, mouth = instances
    calls: list[str] = []

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {"association_score": 1.0})], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        if strategy == "legacy":
            candidate = {
                "message_type": "foam_ring_rim_pinch_grasp_candidate",
                "target": {"ring_instance_id": 0, "mouth_instance_id": 1},
            }
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    raw = _hybrid_raw()
    scene = run_hybrid_grasp(
        instances,
        depth,
        intrinsics,
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        associate_fn=associate,
    )
    assert calls == ["m38_1_front_annulus", "legacy"]
    assert scene["selected_grasp_branch"] == "m36_mouth_visible_rim_pinch"
    assert scene["m38_1_branch_a"]["fallback_to_m36"] is True
