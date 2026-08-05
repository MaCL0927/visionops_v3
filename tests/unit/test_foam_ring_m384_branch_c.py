from __future__ import annotations

from typing import Any

import numpy as np  # type: ignore

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _ring() -> SegmentationInstance:
    mask = np.zeros((60, 100), dtype=bool)
    mask[12:48, 20:72] = True
    return SegmentationInstance(
        instance_id=7,
        class_id=0,
        class_name="foam_ring",
        confidence=0.95,
        mask=mask,
        bbox_xyxy=(20, 12, 72, 48),
    )


def _raw() -> dict[str, Any]:
    return {
        "depth": {"minimum_mm": 150, "maximum_mm": 3000},
        "m38_branch_a": {"enabled": True, "fallback_to_m36": False},
        "m38_branch_b": {"enabled": True, "fallback_to_m36": False},
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


def test_m384_branch_c_fast_rejects_without_invoking_m36_or_m376() -> None:
    ring = _ring()
    depth = np.zeros(ring.mask.shape, dtype=np.uint16)
    depth[ring.mask] = 600
    analyze_calls: list[str] = []
    side_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [], [ring], [], []

    def infer(_ring, _depth, _raw_config):
        return {
            "eligible": False,
            "mouth_instance": None,
            "association": {},
            "rejection_reasons": ["m383_no_depth_partial_opening_component"],
            "diagnostics": {},
            "timing_ms": {"total_ms": 1.0},
            "_debug": {},
        }

    def analyze(*_args, **_kwargs):
        analyze_calls.append("called")
        raise AssertionError("M36/analyze_scene must not run after terminal branch C")

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(int(instance.instance_id))
        raise AssertionError("M37.6 side fitting must not run after terminal branch C")

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        partial_infer_fn=infer,
        analyze_fn=analyze,
        side_fit_fn=side_fit,
    )

    assert analyze_calls == []
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m38_4_branch_c_fast_reject"
    assert scene["robot_candidate"] is None
    assert scene["grasp_result_status"] == "rejected_requires_box_turning"
    assert scene["operator_action"] == "turn_or_agitate_box"
    assert scene["m38_4_branch_c"]["executed"] is True
    assert scene["m38_4_branch_c"]["decision"] == "m38_c_no_accessible_opening_evidence"
    assert scene["m38_4_branch_c"]["legacy_m36_executed"] is False
    assert scene["m38_4_branch_c"]["legacy_m37_6_executed"] is False
    assert scene["hybrid_grasp"]["m36_candidate_found"] is False
    assert scene["hybrid_grasp"]["m37_candidate_found"] is False
    assert scene["hybrid_grasp"]["timing_ms"]["m36_branch_ms"] == 0.0
    assert scene["side_ring_branch"]["fast_attempt_count"] == 0


def test_m384_branch_c_reports_opening_seen_but_no_safe_grasp() -> None:
    ring = _ring()
    mouth_mask = np.zeros(ring.mask.shape, dtype=bool)
    mouth_mask[22:36, 36:52] = True
    mouth = SegmentationInstance(
        instance_id=8,
        class_id=1,
        class_name="ring_mouth",
        confidence=0.94,
        mask=mouth_mask,
        bbox_xyxy=(36, 22, 52, 36),
    )
    depth = np.zeros(ring.mask.shape, dtype=np.uint16)
    depth[ring.mask] = 600

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {"association_mode": "strict_envelope"})], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        assert strategy == "m38_1_front_annulus"
        return {
            "robot_candidate": None,
            "eligible_count": 0,
            "instances": [
                {
                    "ring_instance_id": 7,
                    "mouth_instance_id": 8,
                    "eligible": False,
                    "rejection_reasons": ["full_gripper_static_neighbor_collision"],
                    "warnings": [],
                    "m38_branch_a": {"opening_clear": True, "annulus_point_count": 200},
                    "plane": {"inlier_ratio": 0.8, "residual_p95_mm": 3.0},
                }
            ],
        }

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring, mouth],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
    )

    assert scene["selected_grasp_branch"] == "m38_4_branch_c_fast_reject"
    assert scene["m38_4_branch_c"]["decision"] == "m38_c_no_collision_free_inner_outer_grasp"
