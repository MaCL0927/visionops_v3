"""M37.3 automatic mouth-visible / side-lying foam-ring branch selection.

One Runtime segmentation result and one exact RGB-D frame are shared by both
branches.  The established M36 mouth-visible rim-pinch branch always runs
first.  Only when it produces no valid candidate does the side-ring M37.2
parameterized short-cylinder fit run on unmatched ``foam_ring`` instances in
confidence order, stopping at the first eligible fit.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np  # type: ignore

from .geometry import GeometryConfig, analyze_scene
from .segmentation import SegmentationInstance
from .side_ring_template import SideRingTemplateConfig, fit_side_ring_instance


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


@dataclass(frozen=True)
class HybridGraspConfig:
    """Branch policy for the M37.3 unified trigger path."""

    enabled: bool = False
    prefer_mouth_visible: bool = True
    side_ring_fallback_enabled: bool = True
    side_ring_only_unmatched: bool = True
    side_ring_search_profile: str = "auto"
    stop_after_first_side_eligible: bool = True
    maximum_side_ring_attempts: int = 0

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "HybridGraspConfig":
        section = raw_config.get("hybrid_grasp") or {}
        if not isinstance(section, Mapping):
            raise ValueError("hybrid_grasp must be a mapping")
        profile = str(section.get("side_ring_search_profile") or "auto").strip().lower()
        if profile not in {"auto", "fast", "accurate"}:
            raise ValueError("hybrid_grasp.side_ring_search_profile must be auto, fast or accurate")
        return cls(
            enabled=bool(section.get("enabled", False)),
            prefer_mouth_visible=bool(section.get("prefer_mouth_visible", True)),
            side_ring_fallback_enabled=bool(section.get("side_ring_fallback_enabled", True)),
            side_ring_only_unmatched=bool(section.get("side_ring_only_unmatched", True)),
            side_ring_search_profile=profile,
            stop_after_first_side_eligible=bool(
                section.get("stop_after_first_side_eligible", True)
            ),
            maximum_side_ring_attempts=max(
                0, int(section.get("maximum_side_ring_attempts", 0))
            ),
        )


def _deferred_side_record(
    instance: SegmentationInstance,
    *,
    attempt_rank: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "ring_instance_id": int(instance.instance_id),
        "ring_confidence": float(instance.confidence),
        "ring_bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "mouth_matched": False,
        "attempt_rank": int(attempt_rank),
        "processing_status": "deferred",
        "deferred_reason": str(reason),
        "eligible": None,
        "rejection_reasons": [],
        "timing_ms": {"total_ms": 0.0},
    }


def _m37_candidate(fit: Mapping[str, Any]) -> Dict[str, Any]:
    crown = fit.get("near_side_crown") if isinstance(fit.get("near_side_crown"), Mapping) else {}
    return {
        "schema_version": "1.0",
        "message_type": "foam_ring_side_crown_grasp_candidate",
        "status": "candidate_only_not_robot_ready",
        "robot_ready": False,
        "reason": (
            "M37.3 side-ring camera-frame grasp point only; gripper pose, hand-eye "
            "transform, reachability and final robot protocol are not enabled"
        ),
        "grasp_branch": "m37_side_ring_near_visible_crown",
        "grasp_mode": "side_ring_near_visible_crown",
        "target": {
            "ring_instance_id": fit.get("ring_instance_id"),
            "ring_confidence": fit.get("ring_confidence"),
            "attempt_rank": fit.get("attempt_rank"),
            "fit_score": fit.get("fit_score"),
            "search_profile_used": fit.get("search_profile_used"),
            "accurate_fallback_used": fit.get("accurate_fallback_used"),
            "radial_inlier_ratio": fit.get("radial_inlier_ratio"),
            "radial_residual_median_mm": fit.get("radial_residual_median_mm"),
            "radial_residual_p90_mm": fit.get("radial_residual_p90_mm"),
            "observed_axis_span_mm": fit.get("observed_axis_span_mm"),
            "axis_view_angle_deg": fit.get("axis_view_angle_deg"),
        },
        "grasp_point_camera_mm": crown.get("grasp_point_camera_mm"),
        "grasp_point_uv": crown.get("grasp_point_uv"),
        "axis_toward_camera": fit.get("axis_toward_camera"),
        "center_camera_mm": fit.get("center_camera_mm"),
        "near_opening_center_camera_mm": fit.get("near_opening_center_camera_mm"),
        "far_opening_center_camera_mm": fit.get("far_opening_center_camera_mm"),
        "near_side_crown": deepcopy(crown),
        "fit_timing_ms": deepcopy(fit.get("timing_ms") or {}),
    }


def _m36_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    document = deepcopy(dict(candidate))
    document["grasp_branch"] = "m36_mouth_visible_rim_pinch"
    document["grasp_mode"] = "rim_pinch"
    return document


def run_hybrid_grasp(
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    raw_config: Mapping[str, Any],
    geometry_config: GeometryConfig,
    analyze_fn: Callable[..., Dict[str, Any]] = analyze_scene,
    side_fit_fn: Callable[..., Dict[str, Any]] = fit_side_ring_instance,
) -> Dict[str, Any]:
    """Run M36 first and M37 only when M36 has no valid candidate.

    The returned object is the scene document consumed by the existing online
    processor.  Existing M36 scene keys remain at the top level for backward
    compatibility; M37.3 adds ``hybrid_grasp`` and ``side_ring_branch``.
    """

    hybrid_config = HybridGraspConfig.from_mapping(raw_config)
    if not hybrid_config.enabled:
        return analyze_fn(instances, depth_mm, intrinsics, geometry_config)

    total_started = time.perf_counter()
    m36_started = time.perf_counter()
    m36_scene = analyze_fn(instances, depth_mm, intrinsics, geometry_config)
    m36_ms = _elapsed_ms(m36_started)
    if not isinstance(m36_scene, dict):
        raise ValueError("M36 analyze_scene must return a dict")

    original_m36_candidate = (
        m36_scene.get("robot_candidate")
        if isinstance(m36_scene.get("robot_candidate"), Mapping)
        else None
    )
    m36_candidate = _m36_candidate(original_m36_candidate) if original_m36_candidate else None

    side_fits: List[Dict[str, Any]] = []
    selected_side: Dict[str, Any] | None = None
    candidate_filter_sort_ms = 0.0
    side_fit_loop_ms = 0.0
    selected_branch = "m36_mouth_visible_rim_pinch" if m36_candidate else "none"
    fallback_reason: str | None = None

    if m36_candidate is None and hybrid_config.side_ring_fallback_enabled:
        fallback_reason = "m36_no_valid_candidate"
        filter_started = time.perf_counter()
        unmatched_ids_raw = m36_scene.get("unmatched_ring_ids") or []
        unmatched_ids = {int(value) for value in unmatched_ids_raw}
        side_candidates = [
            item
            for item in instances
            if item.class_name == "foam_ring"
            and (
                not hybrid_config.side_ring_only_unmatched
                or int(item.instance_id) in unmatched_ids
            )
        ]
        side_candidates.sort(
            key=lambda item: (-float(item.confidence), int(item.instance_id))
        )
        candidate_filter_sort_ms = _elapsed_ms(filter_started)

        template_config = SideRingTemplateConfig.from_mapping(raw_config)
        fit_started = time.perf_counter()
        stop_index: int | None = None
        for index, instance in enumerate(side_candidates):
            if (
                hybrid_config.maximum_side_ring_attempts > 0
                and index >= hybrid_config.maximum_side_ring_attempts
            ):
                stop_index = index
                break
            fit = side_fit_fn(
                instance,
                depth_mm,
                intrinsics,
                template_config,
                mouth_matched=False,
                search_profile=hybrid_config.side_ring_search_profile,
            )
            fit["attempt_rank"] = int(index + 1)
            fit["processing_status"] = "evaluated"
            side_fits.append(fit)
            if bool(fit.get("eligible", False)):
                selected_side = fit
                if hybrid_config.stop_after_first_side_eligible:
                    stop_index = index + 1
                    break
        side_fit_loop_ms = _elapsed_ms(fit_started)

        if stop_index is not None and stop_index < len(side_candidates):
            reason = (
                "after_first_valid_confidence_candidate"
                if selected_side is not None
                else "maximum_side_ring_attempts_reached"
            )
            for index in range(stop_index, len(side_candidates)):
                side_fits.append(
                    _deferred_side_record(
                        side_candidates[index],
                        attempt_rank=index + 1,
                        reason=reason,
                    )
                )

        if selected_side is not None:
            selected_branch = "m37_side_ring_near_visible_crown"

    selected_candidate = m36_candidate or (
        _m37_candidate(selected_side) if selected_side is not None else None
    )
    evaluated_side = [
        fit for fit in side_fits if fit.get("processing_status") == "evaluated"
    ]
    deferred_side = [
        fit for fit in side_fits if fit.get("processing_status") == "deferred"
    ]
    selected_side_id = (
        int(selected_side.get("ring_instance_id")) if selected_side is not None else None
    )
    total_ms = _elapsed_ms(total_started)

    result = dict(m36_scene)
    result["m36_eligible_count"] = m36_scene.get("eligible_count")
    if selected_side is not None:
        result["eligible_count"] = 1
        result["selected_ring_instance_id"] = selected_side_id
        result["selected_clock_hour"] = None
        result["selected_clock_angle_deg_cw_from_12"] = None
        result["selected_clock_search_batch"] = None
    # Preserve the unmodified M36 candidate for diagnostics and expose one
    # branch-neutral candidate at the historical robot_candidate key.
    result["m36_robot_candidate"] = deepcopy(original_m36_candidate)
    result["robot_candidate"] = selected_candidate
    result["selected_grasp_branch"] = selected_branch
    result["hybrid_grasp"] = {
        "enabled": True,
        "branch_priority": [
            "m36_mouth_visible_rim_pinch",
            "m37_side_ring_near_visible_crown",
        ],
        "selected_branch": selected_branch,
        "fallback_triggered": bool(m36_candidate is None),
        "fallback_reason": fallback_reason,
        "m36_candidate_found": bool(m36_candidate is not None),
        "m37_candidate_found": bool(selected_side is not None),
        "target_found": bool(selected_candidate is not None),
        "timing_ms": {
            "m36_branch_ms": float(m36_ms),
            "m37_candidate_filter_sort_ms": float(candidate_filter_sort_ms),
            "m37_fit_loop_ms": float(side_fit_loop_ms),
            "m37_evaluated_instance_total_ms": float(
                sum(
                    float((fit.get("timing_ms") or {}).get("total_ms", 0.0))
                    for fit in evaluated_side
                )
            ),
            "total_ms": float(total_ms),
        },
    }
    result["side_ring_branch"] = {
        "enabled": bool(hybrid_config.side_ring_fallback_enabled),
        "executed": bool(m36_candidate is None and hybrid_config.side_ring_fallback_enabled),
        "candidate_order_rule": "foam_ring_confidence_descending",
        "only_unmatched_m36_rings": bool(hybrid_config.side_ring_only_unmatched),
        "search_profile": hybrid_config.side_ring_search_profile,
        "candidate_count": int(len(evaluated_side) + len(deferred_side)),
        "evaluated_count": int(len(evaluated_side)),
        "deferred_count": int(len(deferred_side)),
        "selected_ring_instance_id": selected_side_id,
        "fits": side_fits,
    }
    return result
