from __future__ import annotations

from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import GeometryConfig
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import run_hybrid_grasp
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.partial_opening_cylinder_m383 import (
    infer_depth_partial_opening,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import SegmentationInstance


def _instance(instance_id: int, class_name: str, mask: np.ndarray) -> SegmentationInstance:
    ys, xs = np.nonzero(mask)
    return SegmentationInstance(
        instance_id=instance_id,
        class_id=0 if class_name == "foam_ring" else 1,
        class_name=class_name,
        confidence=0.95,
        mask=mask.astype(bool),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )


def _raw() -> dict[str, Any]:
    return {
        "association": {
            "minimum_mouth_to_ring_area_ratio": 0.025,
            "maximum_mouth_to_ring_area_ratio": 0.70,
        },
        "depth": {"minimum_mm": 150, "maximum_mm": 3000},
        "object_geometry": {
            "nominal_outer_diameter_mm": 85.0,
            "nominal_inner_diameter_mm": 60.0,
            "axial_length_mm": 70.0,
        },
        "m38_branch_a": {"enabled": True, "fallback_to_m36": True},
        "m38_branch_b": {
            "enabled": True,
            "fallback_to_m36": True,
            "maximum_candidates_per_trigger": 4,
            "minimum_partial_opening_center_offset_ratio": 0.10,
            "inferred_opening_ring_erode_px": 1,
            "inferred_opening_minimum_depth_gap_mm": 35.0,
            "inferred_opening_minimum_area_ratio": 0.04,
            "inferred_opening_maximum_area_ratio": 0.36,
            "inferred_opening_minimum_center_offset_ratio": 0.12,
            "inferred_opening_maximum_center_offset_ratio": 0.68,
            "inferred_opening_maximum_boundary_contact_ratio": 0.40,
            "inferred_opening_minimum_inside_distance_px": 3.0,
            "inferred_opening_rim_dilate_px": 4,
            "inferred_opening_minimum_rim_support_px": 10,
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


def test_m383_depth_infers_off_center_aperture_inside_unmatched_ring() -> None:
    shape = (100, 140)
    ring_mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(ring_mask, (70, 50), (45, 34), 0, 0, 360, 1, -1)
    ring = _instance(2, "foam_ring", ring_mask.astype(bool))
    depth = np.zeros(shape, dtype=np.uint16)
    depth[ring.mask] = 600
    aperture = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(aperture, (48, 66), (14, 10), -20, 0, 360, 1, -1)
    aperture = aperture.astype(bool) & ring.mask
    depth[aperture] = 680

    evidence = infer_depth_partial_opening(ring, depth, _raw())
    assert evidence["eligible"] is True
    assert evidence["diagnostics"]["opening_source"] == "depth_inferred"
    assert evidence["diagnostics"]["opening_depth_gap_mm"] >= 70.0
    assert evidence["diagnostics"]["opening_center_offset_ratio"] >= 0.12
    assert isinstance(evidence["mouth_instance"], SegmentationInstance)


def test_m383_unmatched_depth_opening_enters_branch_b_before_legacy() -> None:
    shape = (60, 100)
    ring_mask = np.zeros(shape, dtype=bool)
    ring_mask[10:50, 20:70] = True
    mouth_mask = np.zeros(shape, dtype=bool)
    mouth_mask[28:42, 22:38] = True
    ring = _instance(2, "foam_ring", ring_mask)
    inferred_mouth = _instance(-100002, "ring_mouth", mouth_mask)
    depth = np.zeros(shape, dtype=np.uint16)
    depth[ring.mask] = 600
    calls: list[str] = []

    def associate(_rings, _mouths, _config):
        return [], [ring], [], []

    def infer(_ring, _depth, _raw_config):
        return {
            "eligible": True,
            "mouth_instance": inferred_mouth,
            "association": {
                "association_mode": "depth_inferred_partial_opening",
                "containment": 1.0,
                "mouth_to_ring_area_ratio": 0.11,
                "association_score": 4.0,
            },
            "rejection_reasons": [],
            "diagnostics": {"opening_source": "depth_inferred", "evidence_score": 4.0},
            "timing_ms": {"total_ms": 1.0},
            "_debug": {},
        }

    def fit(*_args, **_kwargs):
        return {
            "ring_instance_id": 2,
            "mouth_instance_id": -100002,
            "eligible": True,
            "rejection_reasons": [],
            "warnings": [],
            "association": {"association_mode": "depth_inferred_partial_opening"},
            "diagnostics": {"opening_source": "depth_inferred"},
            "pose_payload": {
                "ring_instance_id": 2,
                "mouth_instance_id": -100002,
                "normal_toward_camera": [0.8, 0.0, -0.6],
                "opening_center_camera_mm": [0.0, 0.0, 600.0],
                "far_opening_center_camera_mm": [-56.0, 0.0, 642.0],
                "side_point_count": 200,
                "side_plane_inlier_ratio": 0.8,
                "side_residual_median_mm": 2.0,
                "side_residual_p95_mm": 5.0,
                "diagnostics": {},
            },
            "synthetic_mouth_instance": inferred_mouth,
            "synthetic_ring_instance": ring,
            "timing_ms": {"total_ms": 10.0},
        }

    def analyze(_instances, _depth, _intrinsics, config):
        strategy = str(config.section("_runtime").get("pose_strategy") or "legacy")
        calls.append(strategy)
        candidate = None
        if strategy == "m38_2_partial_opening_cylinder":
            candidate = {"target": {"ring_instance_id": 2, "mouth_instance_id": -100002}}
        return {"robot_candidate": candidate, "eligible_count": int(candidate is not None), "instances": []}

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        partial_infer_fn=infer,
        partial_fit_fn=fit,
    )
    assert calls == ["m38_2_partial_opening_cylinder"]
    assert scene["selected_grasp_branch"] == "m38_3_partial_opening_constrained_cylinder_rim_pinch"
    assert scene["robot_candidate"]["pose_source"] == "depth_or_segmented_partial_opening_constrained_cylinder"
    assert scene["m38_3_branch_b"]["preselection_sources"] == {"2": "depth_inferred_partial_opening"}
    assert scene["m38_2_branch_b"]["deprecated_alias_for"] == "m38_3_branch_b"


def test_m383_concentric_segmented_mouth_is_not_reused_as_branch_b_evidence() -> None:
    shape = (60, 100)
    ring_mask = np.zeros(shape, dtype=bool)
    ring_mask[10:50, 25:75] = True
    mouth_mask = np.zeros(shape, dtype=bool)
    mouth_mask[22:38, 42:58] = True
    ring = _instance(0, "foam_ring", ring_mask)
    mouth = _instance(1, "ring_mouth", mouth_mask)
    depth = np.zeros(shape, dtype=np.uint16)
    depth[ring.mask] = 600
    fit_calls: list[int] = []

    def associate(_rings, _mouths, _config):
        return [(ring, mouth, {
            "association_mode": "strict_envelope",
            "containment": 1.0,
            "mouth_to_ring_area_ratio": 0.12,
        })], [], [], []

    def analyze(_instances, _depth, _intrinsics, config):
        return {"robot_candidate": None, "eligible_count": 0, "instances": []}

    raw = _raw()
    scene = run_hybrid_grasp(
        [ring, mouth],
        depth,
        {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 30.0},
        raw_config=raw,
        geometry_config=GeometryConfig(raw),
        associate_fn=associate,
        analyze_fn=analyze,
        partial_fit_fn=lambda *_args, **_kwargs: fit_calls.append(1),
    )
    assert fit_calls == []
    assert scene["m38_3_branch_b"]["attempt_count"] == 0
    assert scene["m38_3_branch_b"]["preselected_ring_instance_ids"] == []
