from __future__ import annotations

from typing import Any

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _ring(instance_id: int, confidence: float) -> SegmentationInstance:
    mask = np.zeros((32, 32), dtype=bool)
    mask[5:25, 5:25] = True
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=0,
        class_name="foam_ring",
        confidence=confidence,
        mask=mask,
        bbox_xyxy=(5, 5, 25, 25),
    )


def _config() -> dict[str, Any]:
    return {
        "hybrid_grasp": {
            "enabled": True,
            "side_ring_fallback_enabled": True,
            "side_ring_only_unmatched": True,
            "side_ring_search_profile": "auto",
            "stop_after_first_side_eligible": True,
        },
        "side_ring_template": {},
    }


def test_m36_candidate_has_priority_and_skips_m37():
    raw = _config()
    side_calls: list[int] = []

    def analyze(*_args):
        return {
            "rings_detected": 1,
            "mouths_detected": 1,
            "matched_pairs": 1,
            "unmatched_ring_ids": [],
            "robot_candidate": {"message_type": "foam_ring_rim_pinch_grasp_candidate"},
        }

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(instance.instance_id)
        raise AssertionError("M37 must not run when M36 has a candidate")

    scene = run_hybrid_grasp(
        [_ring(1, 0.9)],
        np.full((32, 32), 500, dtype=np.uint16),
        {"fx": 600.0, "fy": 600.0, "cx": 16.0, "cy": 16.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
    )
    assert side_calls == []
    assert scene["selected_grasp_branch"] == "m36_mouth_visible_rim_pinch"
    assert scene["robot_candidate"]["grasp_branch"] == "m36_mouth_visible_rim_pinch"
    assert scene["hybrid_grasp"]["fallback_triggered"] is False


def test_m37_fallback_sorts_unmatched_by_confidence_and_stops_first_success():
    raw = _config()
    rings = [_ring(1, 0.80), _ring(2, 0.99), _ring(3, 0.95)]
    side_calls: list[int] = []

    def analyze(*_args):
        return {
            "rings_detected": 3,
            "mouths_detected": 1,
            "matched_pairs": 1,
            # Ring 2 is mouth-matched and must not enter M37 despite highest confidence.
            "unmatched_ring_ids": [1, 3],
            "robot_candidate": None,
        }

    def side_fit(instance, *_args, **_kwargs):
        side_calls.append(instance.instance_id)
        eligible = instance.instance_id == 1
        return {
            "ring_instance_id": instance.instance_id,
            "ring_confidence": instance.confidence,
            "eligible": eligible,
            "rejection_reasons": [] if eligible else ["synthetic_failure"],
            "fit_score": 1.0,
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

    scene = run_hybrid_grasp(
        rings,
        np.full((32, 32), 500, dtype=np.uint16),
        {"fx": 600.0, "fy": 600.0, "cx": 16.0, "cy": 16.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        analyze_fn=analyze,
        side_fit_fn=side_fit,
    )
    # Unmatched rings are confidence-sorted: ring 3 fails, then ring 1 succeeds.
    assert side_calls == [3, 1]
    assert scene["selected_grasp_branch"] == "m37_side_ring_near_visible_crown"
    assert scene["robot_candidate"]["target"]["ring_instance_id"] == 1
    assert scene["robot_candidate"]["grasp_point_uv"] == [100.0, 120.0]
    assert scene["side_ring_branch"]["evaluated_count"] == 2
    assert scene["hybrid_grasp"]["fallback_triggered"] is True
