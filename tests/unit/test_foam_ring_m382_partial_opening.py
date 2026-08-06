from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np  # type: ignore

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (
    HybridGraspConfig,
    run_hybrid_grasp,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import load_yaml
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


REPO_ROOT = Path(__file__).resolve().parents[2]
LINE_CONFIG = REPO_ROOT / "production/foam_ring_grasp/config/line.yaml"


def _instance(instance_id: int, class_id: int, class_name: str, x1: int, depth_shape=(60, 100)) -> SegmentationInstance:
    mask = np.zeros(depth_shape, dtype=bool)
    if class_name == "foam_ring":
        mask[12:48, x1 : x1 + 32] = True
    else:
        mask[20:38, x1 : x1 + 14] = True
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.95,
        mask=mask,
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _raw() -> dict[str, Any]:
    return {
        "m38_branch_a": {"enabled": True, "fallback_to_m36": True},
        "m38_branch_b": {
            "enabled": True,
            "fallback_to_m36": True,
            "maximum_candidates_per_trigger": 4,
        },
        "hybrid_grasp": {
            "enabled": True,
            "prefer_mouth_visible": True,
            "side_ring_fallback_enabled": True,
            "side_ring_only_unmatched": True,
            "multi_surface_include_m36_rejected": True,
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


def _fit_payload(ring: SegmentationInstance, mouth: SegmentationInstance) -> dict[str, Any]:
    return {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "eligible": True,
        "rejection_reasons": [],
        "warnings": [],
        "association": {"association_mode": "bbox_fallback"},
        "diagnostics": {
            "pose_source": "depth_or_segmented_partial_opening_constrained_cylinder",
            "radial_inlier_ratio": 0.85,
            "axis_view_angle_deg": 62.0,
        },
        "pose_payload": {
            "ring_instance_id": int(ring.instance_id),
            "mouth_instance_id": int(mouth.instance_id),
            "normal_toward_camera": [0.8, 0.0, -0.6],
            "opening_center_camera_mm": [0.0, 0.0, 600.0],
            "far_opening_center_camera_mm": [-56.0, 0.0, 642.0],
            "side_point_count": 240,
            "side_plane_inlier_ratio": 0.85,
            "side_residual_median_mm": 2.0,
            "side_residual_p95_mm": 5.0,
            "diagnostics": {
                "pose_source": "depth_or_segmented_partial_opening_constrained_cylinder",
                "axis_view_angle_deg": 62.0,
            },
        },
        "synthetic_mouth_instance": mouth,
        "synthetic_ring_instance": ring,
        "timing_ms": {"total_ms": 15.0},
    }


def test_m382_production_configuration_enables_branch_b_and_m384_terminal_c() -> None:
    raw = load_yaml(LINE_CONFIG)
    hybrid = HybridGraspConfig.from_mapping(raw)
    assert raw["schema_version"] == "6.4"
    assert raw["task"] == "foam_ring_rim_pinch_m38_6_direction_collision_contact_fix"
    assert hybrid.m38_branch_a_enabled is True
    assert hybrid.m38_branch_b_enabled is True
    assert hybrid.m38_branch_b_fallback_to_m36 is False
    assert hybrid.m38_branch_b_maximum_candidates == 4
    assert hybrid.m38_branch_c_enabled is True
    assert hybrid.m38_branch_c_fast_terminate is True
    assert hybrid.legacy_m36_enabled is False
    assert raw["hybrid_grasp"]["side_ring_fallback_enabled"] is False


def test_m382_branch_a_still_has_priority_and_skips_branch_b() -> None:
    ring = _instance(0, 0, "foam_ring", 15)
    mouth = _instance(1, 1, "ring_mouth", 38)
    depth = np.zeros(ring.mask.shape, dtype=np.uint16)
    depth[ring.mask] = 600
    calls: list[str] = []
    fit_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {"association_mode": "strict_envelope"})], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        if strategy == "m38_1_front_annulus":
            candidate = {"target": {"ring_instance_id": 0, "mouth_instance_id": 1}}
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    def partial_fit(*_args, **_kwargs):
        fit_calls.append(0)
        return _fit_payload(ring, mouth)

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring, mouth], depth, {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        partial_fit_fn=partial_fit,
    )
    assert calls == ["m38_1_front_annulus"]
    assert fit_calls == []
    assert scene["selected_grasp_branch"] == "m38_1_clear_mouth_front_annulus_rim_pinch"
    assert scene["hybrid_grasp"]["policy_version"] == "M38.6"


def test_m382_uses_partial_opening_cylinder_before_m36_and_m376() -> None:
    ring = _instance(0, 0, "foam_ring", 15)
    mouth = _instance(1, 1, "ring_mouth", 38)
    depth = np.zeros(ring.mask.shape, dtype=np.uint16)
    depth[ring.mask] = 600
    calls: list[str] = []
    side_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        metrics = {
            "association_mode": "bbox_fallback",
            "containment": 0.55,
            "mouth_to_ring_area_ratio": 0.12,
        }
        return [(ring, mouth, metrics)], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        instances = []
        if strategy == "m38_2_partial_opening_cylinder":
            candidate = {"target": {"ring_instance_id": 0, "mouth_instance_id": 1}}
            instances = [{"ring_instance_id": 0, "m38_branch_b": {"opening_partial": True}}]
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": instances}

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(int(instance.instance_id))
        return {"ring_instance_id": int(instance.instance_id), "eligible": False}

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring, mouth], depth, {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        partial_fit_fn=lambda *_args, **_kwargs: _fit_payload(ring, mouth),
    )
    assert calls == ["m38_1_front_annulus", "m38_2_partial_opening_cylinder"]
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m38_3_partial_opening_constrained_cylinder_rim_pinch"
    assert scene["robot_candidate"]["pose_source"] == "depth_or_segmented_partial_opening_constrained_cylinder"
    assert scene["m38_3_branch_b"]["candidate_found"] is True
    assert scene["m38_3_branch_b"]["legacy_m36_retained"] is True
    assert scene["m38_3_branch_b"]["m37_6_retained"] is True


def test_m382_failed_a_and_b_fall_back_to_legacy_m36() -> None:
    ring = _instance(0, 0, "foam_ring", 15)
    mouth = _instance(1, 1, "ring_mouth", 38)
    depth = np.zeros(ring.mask.shape, dtype=np.uint16)
    depth[ring.mask] = 600
    calls: list[str] = []

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {
            "association_mode": "bbox_fallback",
            "containment": 0.55,
            "mouth_to_ring_area_ratio": 0.12,
        })], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        if strategy == "legacy":
            candidate = {"target": {"ring_instance_id": 0, "mouth_instance_id": 1}}
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    rejected = _fit_payload(ring, mouth)
    rejected["eligible"] = False
    rejected["pose_payload"] = None
    rejected["synthetic_mouth_instance"] = None
    rejected["synthetic_ring_instance"] = None
    rejected["rejection_reasons"] = ["m38b_insufficient_partial_end_support"]
    raw = _raw()
    scene = run_hybrid_grasp(
        [ring, mouth], depth, {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        partial_fit_fn=lambda *_args, **_kwargs: rejected,
    )
    assert calls == ["m38_1_front_annulus", "legacy"]
    assert scene["selected_grasp_branch"] == "m36_mouth_visible_rim_pinch"
    assert scene["m38_3_branch_b"]["fit_results"][0]["eligible"] is False


def test_m382_global_m38_search_prevents_shallow_m376_from_delaying_deeper_opening() -> None:
    shallow = _instance(0, 0, "foam_ring", 5)
    deep = _instance(2, 0, "foam_ring", 55)
    mouth = _instance(3, 1, "ring_mouth", 63)
    depth = np.zeros(shallow.mask.shape, dtype=np.uint16)
    depth[shallow.mask] = 500
    depth[deep.mask] = 590
    side_calls: list[int] = []
    scopes: list[list[int]] = []

    def associate(_rings, _mouths, _config):
        return [(deep, mouth, {"association_mode": "strict_envelope"})], [shallow], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        scope = list(config.section("candidate_scope").get("allowed_ring_instance_ids") or [])
        scopes.append(scope)
        candidate = None
        if strategy == "m38_1_front_annulus":
            candidate = {"target": {"ring_instance_id": 2, "mouth_instance_id": 3}}
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(int(instance.instance_id))
        return {"ring_instance_id": int(instance.instance_id), "eligible": True}

    raw = _raw()
    scene = run_hybrid_grasp(
        [shallow, deep, mouth], depth, {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        side_fit_fn=side_fit,
    )
    assert scopes == [[2]]
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m38_1_clear_mouth_front_annulus_rim_pinch"
    assert scene["robot_candidate"]["target"]["depth_layer_index"] == 1
