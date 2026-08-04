from __future__ import annotations

from typing import Any

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _ring(instance_id: int, confidence: float, x1: int = 5) -> SegmentationInstance:
    mask = np.zeros((40, 60), dtype=bool)
    mask[5:25, x1 : x1 + 18] = True
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=0,
        class_name="foam_ring",
        confidence=confidence,
        mask=mask,
        bbox_xyxy=(x1, 5, x1 + 18, 25),
    )


def _mouth(instance_id: int, x1: int = 5) -> SegmentationInstance:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:18, x1 + 5 : x1 + 13] = True
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=1,
        class_name="ring_mouth",
        confidence=0.9,
        mask=mask,
        bbox_xyxy=(x1 + 5, 10, x1 + 13, 18),
    )


def _config() -> dict[str, Any]:
    return {
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
                "maximum_accurate_refinements_per_trigger": 1,
            },
        },
        "side_ring_template": {"fast_accept_max_score": 3.0},
    }


def _fit_payload(instance: SegmentationInstance, *, accepted: bool, score: float = 1.0):
    return {
        "ring_instance_id": int(instance.instance_id),
        "ring_confidence": float(instance.confidence),
        "eligible": bool(accepted),
        "rejection_reasons": [] if accepted else ["synthetic_uncertain"],
        "fit_score": float(score),
        "fast_acceptance_passed": bool(accepted),
        "fast_acceptance_reasons": [] if accepted else ["synthetic_uncertain"],
        "search_profile_used": "fast",
        "accurate_fallback_used": False,
        "radial_inlier_ratio": 0.9,
        "radial_residual_median_mm": 1.0,
        "radial_residual_p90_mm": 2.0,
        "observed_axis_span_mm": 60.0,
        "axis_view_angle_deg": 75.0,
        "axis_toward_camera": [1.0, 0.0, 0.0],
        "center_camera_mm": [0.0, 0.0, 600.0],
        "near_opening_center_camera_mm": [0.0, 0.0, 580.0],
        "far_opening_center_camera_mm": [0.0, 0.0, 650.0],
        "near_side_crown": {
            "grasp_point_camera_mm": [10.0, 20.0, 580.0],
            "grasp_point_uv": [100.0, 120.0],
        },
        "timing_ms": {"total_ms": 10.0},
    }


def test_m374_same_layer_prefers_m36_before_m37():
    raw = _config()
    ring_side = _ring(1, 0.99, 5)
    ring_m36 = _ring(2, 0.80, 32)
    mouth = _mouth(20, 32)
    depth = np.zeros((40, 60), dtype=np.uint16)
    depth[ring_side.mask] = 500
    depth[ring_m36.mask] = 510
    side_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [(ring_m36, mouth, {"association_score": 1.0})], [ring_side], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        allowed = config.section("candidate_scope").get("allowed_ring_instance_ids")
        candidate = None
        if allowed == [2]:
            candidate = {
                "message_type": "foam_ring_rim_pinch_grasp_candidate",
                "target": {"ring_instance_id": 2},
            }
        return {
            "rings_detected": 2,
            "mouths_detected": 1,
            "matched_pairs": 1 if allowed == [2] else 0,
            "unmatched_ring_ids": [1],
            "robot_candidate": candidate,
            "eligible_count": 1 if candidate else 0,
        }

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(instance.instance_id)
        return _fit_payload(instance, accepted=True)

    scene = run_hybrid_grasp(
        [ring_side, ring_m36, mouth],
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 30.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m36_mouth_visible_rim_pinch"
    assert scene["depth_layering"]["selected_layer_index"] == 0
    assert scene["robot_candidate"]["target"]["surface_depth_mm"] == 510.0


def test_m374_shallower_m37_precedes_deeper_m36():
    raw = _config()
    shallow = _ring(1, 0.60, 5)
    deep = _ring(2, 0.99, 32)
    mouth = _mouth(20, 32)
    depth = np.zeros((40, 60), dtype=np.uint16)
    depth[shallow.mask] = 500
    depth[deep.mask] = 580
    analyzed_scopes: list[list[int]] = []

    def associate(_rings, _mouths, _config):
        return [(deep, mouth, {"association_score": 1.0})], [shallow], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        allowed = list(config.section("candidate_scope").get("allowed_ring_instance_ids") or [])
        analyzed_scopes.append(allowed)
        return {
            "rings_detected": 2,
            "mouths_detected": 1,
            "matched_pairs": 0,
            "unmatched_ring_ids": [1],
            "robot_candidate": None,
            "eligible_count": 0,
        }

    side_profiles: list[str] = []

    def side_fit(instance, *_args, **kwargs):
        profile = kwargs["search_profile"]
        side_profiles.append(profile)
        assert profile in {"screen", "final_verify"}
        payload = _fit_payload(instance, accepted=True, score=4.2)
        payload["search_profile_used"] = profile
        payload["preliminary_pose_safe"] = profile == "screen"
        payload["final_pose_safe"] = profile == "final_verify"
        return payload

    scene = run_hybrid_grasp(
        [shallow, deep, mouth],
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 30.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert side_profiles == ["screen", "final_verify"]
    assert scene["selected_grasp_branch"] == "m37_side_ring_near_visible_crown"
    assert scene["robot_candidate"]["target"]["ring_instance_id"] == 1
    assert scene["robot_candidate"]["target"]["depth_layer_index"] == 0
    # The deep M36 scope is never evaluated; [] is only the cheap base scene.
    assert [2] not in analyzed_scopes


def test_m374_side_candidates_follow_depth_not_confidence():
    raw = _config()
    shallow_low_conf = _ring(1, 0.60, 5)
    deeper_high_conf = _ring(2, 0.99, 32)
    depth = np.zeros((40, 60), dtype=np.uint16)
    depth[shallow_low_conf.mask] = 500
    depth[deeper_high_conf.mask] = 510
    calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [], [shallow_low_conf, deeper_high_conf], [], []

    def analyze(*_args):
        return {"robot_candidate": None, "eligible_count": 0}

    def side_fit(instance, *_args, **_kwargs):
        calls.append(int(instance.instance_id))
        return _fit_payload(instance, accepted=True)

    scene = run_hybrid_grasp(
        [shallow_low_conf, deeper_high_conf],
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 30.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert calls == [1, 1]
    assert scene["robot_candidate"]["target"]["ring_instance_id"] == 1
    assert scene["depth_layering"]["selected_depth_rank"] == 1


def test_m374_uses_only_one_warm_start_local_accurate_refinement():
    raw = _config()
    ring1 = _ring(1, 0.90, 5)
    ring2 = _ring(2, 0.80, 32)
    depth = np.zeros((40, 60), dtype=np.uint16)
    depth[ring1.mask] = 500
    depth[ring2.mask] = 510
    calls: list[tuple[int, str]] = []

    def associate(_rings, _mouths, _config):
        return [], [ring1, ring2], [], []

    def analyze(*_args):
        return {"robot_candidate": None, "eligible_count": 0}

    def side_fit(instance, *_args, **kwargs):
        profile = kwargs["search_profile"]
        calls.append((int(instance.instance_id), profile))
        if profile == "local_accurate":
            payload = _fit_payload(instance, accepted=True, score=1.2)
            payload["search_profile_used"] = "local_accurate"
            return payload
        # Ring 2 is the better fast seed but neither is fast-accepted.
        score = 3.8 if instance.instance_id == 1 else 3.2
        return _fit_payload(instance, accepted=False, score=score)

    scene = run_hybrid_grasp(
        [ring1, ring2],
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 30.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert calls == [(1, "screen"), (2, "screen"), (2, "local_accurate")]
    assert scene["side_ring_branch"]["accurate_refinement_count"] == 1
    assert scene["side_ring_branch"]["global_accurate_search_used"] is False
    assert scene["robot_candidate"]["target"]["ring_instance_id"] == 2



def test_m3751_pose_safety_is_decoupled_from_legacy_fast_score():
    raw = _config()
    ring = _ring(1, 0.95, 5)
    depth = np.zeros((40, 60), dtype=np.uint16)
    depth[ring.mask] = 500
    calls: list[str] = []

    def associate(_rings, _mouths, _config):
        return [], [ring], [], []

    def analyze(*_args):
        return {"robot_candidate": None, "eligible_count": 0}

    def side_fit(instance, *_args, **kwargs):
        profile = kwargs["search_profile"]
        calls.append(profile)
        payload = _fit_payload(instance, accepted=True, score=4.8)
        payload["search_profile_used"] = profile
        payload["fast_acceptance_passed"] = False
        payload["preliminary_pose_safe"] = profile == "screen"
        payload["final_pose_safe"] = profile == "final_verify"
        return payload

    scene = run_hybrid_grasp(
        [ring],
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 30.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    assert calls == ["screen", "final_verify"]
    assert scene["selected_grasp_branch"] == "m37_side_ring_near_visible_crown"
    assert scene["side_ring_branch"]["accurate_refinement_count"] == 0
    assert scene["side_ring_branch"]["final_validation_count"] == 1


def test_m3751_lightweight_preselection_limits_expensive_candidates():
    raw = _config()
    raw["hybrid_grasp"]["lightweight_preselection"] = {
        "enabled": True,
        "maximum_candidates_per_layer": 2,
        "mask_erode_px": 0,
        "sample_stride": 1,
        "minimum_valid_points": 5,
        "minimum_valid_ratio": 0.01,
    }
    rings = [_ring(index, 0.9 - 0.01 * index, 4 + 9 * index) for index in range(1, 5)]
    # Expand canvas for four non-overlapping synthetic rings.
    fixed = []
    for index, ring in enumerate(rings, start=1):
        x1 = 2 + (index - 1) * 14
        fixed.append(_ring(index, ring.confidence, x1))
    rings = fixed
    depth = np.zeros((40, 80), dtype=np.uint16)
    # Rebuild masks to width 80.
    expanded = []
    for ring in rings:
        mask = np.zeros((40, 80), dtype=bool)
        x1 = ring.bbox_xyxy[0]
        mask[5:25, x1:x1+12] = True
        expanded.append(SegmentationInstance(
            instance_id=ring.instance_id,
            class_id=0,
            class_name="foam_ring",
            confidence=ring.confidence,
            mask=mask,
            bbox_xyxy=(x1,5,x1+12,25),
        ))
        depth[mask] = 500 + ring.instance_id
    calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [], list(expanded), [], []

    def analyze(*_args):
        return {"robot_candidate": None, "eligible_count": 0}

    def side_fit(instance, *_args, **kwargs):
        calls.append(int(instance.instance_id))
        return _fit_payload(instance, accepted=False, score=5.0)

    scene = run_hybrid_grasp(
        expanded,
        depth,
        {"fx": 600.0, "fy": 600.0, "cx": 40.0, "cy": 20.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
        associate_fn=associate,
    )
    # Two screen calls plus one bounded local refinement of the best screen seed.
    assert calls[:2] == [1, 2]
    assert len([fit for fit in scene["side_ring_branch"]["fits"] if fit.get("processing_status") != "deferred"]) <= 2
    assert scene["side_ring_branch"]["screen_attempt_count"] == 2
