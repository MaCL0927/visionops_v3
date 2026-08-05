"""M38.1 depth-layer-first hybrid foam-ring grasp selection.

A single Runtime segmentation result and exact RGB-D frame feed all retained
branches. Rings are grouped by robust front-surface depth. Within each layer,
M38.1 branch A first handles clearly visible openings by fitting the directly
observed 3-D front annulus. Legacy M36 remains an optional mouth-visible
fallback, followed by the unchanged M37.6 hollow-cylinder side-ring branch.

M37.6.1 stop-loss behavior remains active: depth-gradient is diagnostic only
and online ``local_accurate`` refinement is disabled.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .geometry import (
    GeometryConfig,
    _associate_ring_mouths_detailed,
    analyze_scene,
)
from .segmentation import SegmentationInstance
from .side_ring_template import SideRingTemplateConfig, fit_side_ring_instance


def _other_ring_exclusion_mask(
    target: SegmentationInstance,
    rings: Sequence[SegmentationInstance],
) -> np.ndarray:
    exclusion = np.zeros_like(target.mask, dtype=bool)
    target_id = int(target.instance_id)
    for ring in rings:
        if int(ring.instance_id) == target_id:
            continue
        exclusion |= ring.mask.astype(bool)
    return exclusion


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class HybridGraspConfig:
    """M37.5 unified depth-layer, bounded-refinement and pose-safety policy."""

    enabled: bool = False
    prefer_mouth_visible: bool = True
    side_ring_fallback_enabled: bool = True
    side_ring_only_unmatched: bool = True
    stop_after_first_side_eligible: bool = True
    maximum_side_ring_attempts: int = 0

    depth_layering_enabled: bool = True
    depth_layer_tolerance_mm: float = 30.0
    surface_depth_percentile: float = 25.0
    surface_depth_mask_erode_px: int = 3
    surface_depth_mouth_exclusion_px: int = 2
    surface_depth_sample_stride: int = 2
    surface_depth_minimum_points: int = 20
    surface_depth_minimum_valid_ratio: float = 0.08
    surface_depth_minimum_mm: float = 150.0
    surface_depth_maximum_mm: float = 3000.0

    m37_fast_first_enabled: bool = True
    maximum_accurate_refinements_per_trigger: int = 0

    # M37.5.1 lightweight preselection and delayed final pose validation.
    lightweight_preselection_enabled: bool = True
    lightweight_maximum_candidates_per_layer: int = 3
    lightweight_mask_erode_px: int = 2
    lightweight_sample_stride: int = 3
    lightweight_neighbor_contact_dilate_px: int = 2
    lightweight_depth_iqr_reference_mm: float = 40.0
    lightweight_depth_edge_threshold_mm: float = 20.0
    lightweight_minimum_valid_points: int = 30
    lightweight_minimum_valid_ratio: float = 0.20
    delayed_final_validation_enabled: bool = True

    # M37.6: after the M36 branch fails, matched-mouth rings remain eligible for
    # hollow-cylinder multi-surface fitting. The mouth mask is an optional face
    # constraint rather than a reason to exclude the ring from M37.6.
    multi_surface_include_m36_rejected: bool = True

    # M38.1 branch A: clear mouth -> direct 3-D front-annulus plane.  M36 and
    # M37.6 remain available as controlled fallbacks during validation.
    m38_branch_a_enabled: bool = False
    m38_branch_a_fallback_to_m36: bool = True

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "HybridGraspConfig":
        section = raw_config.get("hybrid_grasp") or {}
        if not isinstance(section, Mapping):
            raise ValueError("hybrid_grasp must be a mapping")
        depth = section.get("depth_layering") or {}
        if not isinstance(depth, Mapping):
            depth = {}
        bounded = section.get("bounded_refinement") or {}
        if not isinstance(bounded, Mapping):
            bounded = {}
        lightweight = section.get("lightweight_preselection") or {}
        if not isinstance(lightweight, Mapping):
            lightweight = {}
        depth_cfg = raw_config.get("depth") or {}
        if not isinstance(depth_cfg, Mapping):
            depth_cfg = {}
        m38_branch_a = raw_config.get("m38_branch_a") or {}
        if not isinstance(m38_branch_a, Mapping):
            m38_branch_a = {}
        return cls(
            enabled=bool(section.get("enabled", False)),
            prefer_mouth_visible=bool(section.get("prefer_mouth_visible", True)),
            side_ring_fallback_enabled=bool(
                section.get("side_ring_fallback_enabled", True)
            ),
            side_ring_only_unmatched=bool(
                section.get("side_ring_only_unmatched", True)
            ),
            stop_after_first_side_eligible=bool(
                section.get("stop_after_first_side_eligible", True)
            ),
            maximum_side_ring_attempts=max(
                0, _safe_int(section.get("maximum_side_ring_attempts"), 0)
            ),
            depth_layering_enabled=bool(depth.get("enabled", True)),
            depth_layer_tolerance_mm=max(
                1.0, _safe_float(depth.get("layer_tolerance_mm"), 30.0)
            ),
            surface_depth_percentile=min(
                50.0,
                max(1.0, _safe_float(depth.get("surface_percentile"), 25.0)),
            ),
            surface_depth_mask_erode_px=max(
                0, _safe_int(depth.get("mask_erode_px"), 3)
            ),
            surface_depth_mouth_exclusion_px=max(
                0, _safe_int(depth.get("mouth_exclusion_dilate_px"), 2)
            ),
            surface_depth_sample_stride=max(
                1, _safe_int(depth.get("sample_stride"), 2)
            ),
            surface_depth_minimum_points=max(
                5, _safe_int(depth.get("minimum_valid_points"), 20)
            ),
            surface_depth_minimum_valid_ratio=min(
                1.0,
                max(
                    0.0,
                    _safe_float(depth.get("minimum_valid_ratio"), 0.08),
                ),
            ),
            surface_depth_minimum_mm=_safe_float(
                depth.get("minimum_depth_mm"),
                _safe_float(depth_cfg.get("minimum_mm"), 150.0),
            ),
            surface_depth_maximum_mm=_safe_float(
                depth.get("maximum_depth_mm"),
                _safe_float(depth_cfg.get("maximum_mm"), 3000.0),
            ),
            m37_fast_first_enabled=bool(bounded.get("fast_first_enabled", True)),
            maximum_accurate_refinements_per_trigger=max(
                0,
                _safe_int(
                    bounded.get("maximum_accurate_refinements_per_trigger"), 0
                ),
            ),
            lightweight_preselection_enabled=bool(
                lightweight.get("enabled", True)
            ),
            lightweight_maximum_candidates_per_layer=max(
                1, _safe_int(lightweight.get("maximum_candidates_per_layer"), 3)
            ),
            lightweight_mask_erode_px=max(
                0, _safe_int(lightweight.get("mask_erode_px"), 2)
            ),
            lightweight_sample_stride=max(
                1, _safe_int(lightweight.get("sample_stride"), 3)
            ),
            lightweight_neighbor_contact_dilate_px=max(
                0, _safe_int(lightweight.get("neighbor_contact_dilate_px"), 2)
            ),
            lightweight_depth_iqr_reference_mm=max(
                1.0, _safe_float(lightweight.get("depth_iqr_reference_mm"), 40.0)
            ),
            lightweight_depth_edge_threshold_mm=max(
                2.0, _safe_float(lightweight.get("depth_edge_threshold_mm"), 20.0)
            ),
            lightweight_minimum_valid_points=max(
                5, _safe_int(lightweight.get("minimum_valid_points"), 30)
            ),
            lightweight_minimum_valid_ratio=min(
                1.0, max(0.0, _safe_float(lightweight.get("minimum_valid_ratio"), 0.20))
            ),
            delayed_final_validation_enabled=bool(
                lightweight.get("delayed_final_validation_enabled", True)
            ),
            multi_surface_include_m36_rejected=bool(
                section.get("multi_surface_include_m36_rejected", True)
            ),
            m38_branch_a_enabled=bool(m38_branch_a.get("enabled", False)),
            m38_branch_a_fallback_to_m36=bool(
                m38_branch_a.get("fallback_to_m36", True)
            ),
        )


def _kernel(radius: int) -> np.ndarray:
    size = max(1, 2 * int(radius) + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _surface_depth_record(
    ring: SegmentationInstance,
    depth_mm: np.ndarray,
    config: HybridGraspConfig,
    *,
    mouth: Optional[SegmentationInstance] = None,
) -> Dict[str, Any]:
    """Compute a robust near-surface depth inside one ring instance."""

    started = time.perf_counter()
    x1, y1, x2, y2 = [int(value) for value in ring.bbox_xyxy]
    padding = max(
        config.surface_depth_mask_erode_px,
        config.surface_depth_mouth_exclusion_px,
        1,
    )
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(depth_mm.shape[1], x2 + padding)
    y2 = min(depth_mm.shape[0], y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return {
            "ring_instance_id": int(ring.instance_id),
            "surface_depth_mm": None,
            "depth_valid_count": 0,
            "depth_sample_count": 0,
            "depth_valid_ratio": 0.0,
            "depth_status": "invalid_bbox",
            "timing_ms": _elapsed_ms(started),
        }

    local_mask = ring.mask[y1:y2, x1:x2].astype(np.uint8)
    if config.surface_depth_mask_erode_px > 0:
        local_mask = cv2.erode(
            local_mask,
            _kernel(config.surface_depth_mask_erode_px),
            iterations=1,
        )
    if mouth is not None:
        mouth_mask = mouth.mask[y1:y2, x1:x2].astype(np.uint8)
        if config.surface_depth_mouth_exclusion_px > 0:
            mouth_mask = cv2.dilate(
                mouth_mask,
                _kernel(config.surface_depth_mouth_exclusion_px),
                iterations=1,
            )
        local_mask[mouth_mask > 0] = 0

    stride = config.surface_depth_sample_stride
    sampled_mask = local_mask[::stride, ::stride] > 0
    sampled_depth = depth_mm[y1:y2, x1:x2][::stride, ::stride]
    values = sampled_depth[sampled_mask].astype(np.float64, copy=False)
    sample_count = int(values.size)
    valid = values[
        (values >= config.surface_depth_minimum_mm)
        & (values <= config.surface_depth_maximum_mm)
    ]
    valid_count = int(valid.size)
    valid_ratio = float(valid_count) / float(max(1, sample_count))
    enough = (
        valid_count >= config.surface_depth_minimum_points
        and valid_ratio >= config.surface_depth_minimum_valid_ratio
    )
    surface_depth = (
        float(np.percentile(valid, config.surface_depth_percentile))
        if enough
        else None
    )
    return {
        "ring_instance_id": int(ring.instance_id),
        "ring_confidence": float(ring.confidence),
        "surface_depth_mm": surface_depth,
        "surface_depth_statistic": f"p{config.surface_depth_percentile:g}",
        "depth_median_mm": float(np.median(valid)) if valid_count else None,
        "depth_p10_mm": float(np.percentile(valid, 10)) if valid_count else None,
        "depth_p25_mm": float(np.percentile(valid, 25)) if valid_count else None,
        "depth_valid_count": valid_count,
        "depth_sample_count": sample_count,
        "depth_valid_ratio": valid_ratio,
        "depth_status": "ok" if enough else "insufficient_valid_depth",
        "timing_ms": _elapsed_ms(started),
    }



def _lightweight_side_candidate_record(
    ring: SegmentationInstance,
    rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    depth_record: Mapping[str, Any],
    config: HybridGraspConfig,
) -> Dict[str, Any]:
    """Cheap per-instance quality estimate used before any cylinder fit.

    This stage deliberately avoids 3-D circle fitting, Top-K hypotheses and
    bootstrap. It only measures whether the mask contains a coherent, visible
    and weakly contaminated depth surface. The score is a ranking signal; it is
    never a pose-safety acceptance decision.
    """

    started = time.perf_counter()
    x1, y1, x2, y2 = [int(value) for value in ring.bbox_xyxy]
    x1 = max(0, min(depth_mm.shape[1], x1))
    x2 = max(0, min(depth_mm.shape[1], x2))
    y1 = max(0, min(depth_mm.shape[0], y1))
    y2 = max(0, min(depth_mm.shape[0], y2))
    base = {
        "ring_instance_id": int(ring.instance_id),
        "surface_depth_mm": depth_record.get("surface_depth_mm"),
        "depth_layer_index": depth_record.get("depth_layer_index"),
        "depth_rank": depth_record.get("depth_rank"),
        "ring_confidence": float(ring.confidence),
    }
    if x2 <= x1 or y2 <= y1:
        return {
            **base,
            "status": "invalid_bbox",
            "eligible_for_screen": False,
            "score": float("inf"),
            "timing_ms": _elapsed_ms(started),
        }

    local_mask = ring.mask[y1:y2, x1:x2].astype(np.uint8, copy=True)
    if config.lightweight_mask_erode_px > 0:
        local_mask = cv2.erode(
            local_mask,
            _kernel(config.lightweight_mask_erode_px),
            iterations=1,
        )
    retained_count = int(np.count_nonzero(local_mask))
    if retained_count <= 0:
        return {
            **base,
            "status": "empty_eroded_mask",
            "eligible_for_screen": False,
            "score": float("inf"),
            "retained_pixel_count": 0,
            "timing_ms": _elapsed_ms(started),
        }

    other = np.zeros_like(local_mask, dtype=np.uint8)
    for candidate in rings:
        if int(candidate.instance_id) == int(ring.instance_id):
            continue
        other |= candidate.mask[y1:y2, x1:x2].astype(np.uint8)
    if config.lightweight_neighbor_contact_dilate_px > 0 and np.any(other):
        other = cv2.dilate(
            other,
            _kernel(config.lightweight_neighbor_contact_dilate_px),
            iterations=1,
        )
    contact_ratio = float(np.count_nonzero((local_mask > 0) & (other > 0))) / float(
        max(1, retained_count)
    )

    stride = config.lightweight_sample_stride
    sampled_mask = local_mask[::stride, ::stride] > 0
    sampled_depth = depth_mm[y1:y2, x1:x2][::stride, ::stride].astype(np.float64)
    values = sampled_depth[sampled_mask]
    valid = values[
        (values >= config.surface_depth_minimum_mm)
        & (values <= config.surface_depth_maximum_mm)
    ]
    sample_count = int(values.size)
    valid_count = int(valid.size)
    valid_ratio = float(valid_count) / float(max(1, sample_count))

    if valid_count:
        q10, q25, q50, q75, q90 = [
            float(value)
            for value in np.percentile(valid, [10.0, 25.0, 50.0, 75.0, 90.0])
        ]
        depth_iqr = q75 - q25
        depth_p80_span = q90 - q10
    else:
        q10 = q25 = q50 = q75 = q90 = None
        depth_iqr = float("inf")
        depth_p80_span = float("inf")

    # A sparse organized-depth edge statistic catches masks dominated by
    # discontinuities without computing 3-D normals.
    edge_ratio = 1.0
    if np.any(sampled_mask):
        edge = np.zeros_like(sampled_mask, dtype=bool)
        horizontal_valid = sampled_mask[:, 1:] & sampled_mask[:, :-1]
        vertical_valid = sampled_mask[1:, :] & sampled_mask[:-1, :]
        horizontal_jump = np.abs(sampled_depth[:, 1:] - sampled_depth[:, :-1])
        vertical_jump = np.abs(sampled_depth[1:, :] - sampled_depth[:-1, :])
        edge[:, 1:] |= horizontal_valid & (
            horizontal_jump > config.lightweight_depth_edge_threshold_mm
        )
        edge[1:, :] |= vertical_valid & (
            vertical_jump > config.lightweight_depth_edge_threshold_mm
        )
        edge_ratio = float(np.count_nonzero(edge & sampled_mask)) / float(
            max(1, np.count_nonzero(sampled_mask))
        )

    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_fill_ratio = float(retained_count) / float(bbox_area)
    enough = bool(
        valid_count >= config.lightweight_minimum_valid_points
        and valid_ratio >= config.lightweight_minimum_valid_ratio
    )
    iqr_term = min(3.0, depth_iqr / config.lightweight_depth_iqr_reference_mm)
    span_term = min(3.0, depth_p80_span / (2.0 * config.lightweight_depth_iqr_reference_mm))
    # Lower is better. Depth layer remains the hard outer ordering; this score
    # only ranks candidates inside the same physical layer.
    score = (
        2.0 * (1.0 - valid_ratio)
        + 1.15 * iqr_term
        + 0.60 * span_term
        + 2.5 * contact_ratio
        + 2.0 * edge_ratio
        + 0.35 * max(0.0, 0.25 - mask_fill_ratio)
        - 0.15 * float(ring.confidence)
    )
    return {
        **base,
        "status": "ok" if enough else "insufficient_depth_evidence",
        "eligible_for_screen": enough,
        "score": float(score) if enough else float("inf"),
        "sample_count": sample_count,
        "valid_count": valid_count,
        "valid_ratio": float(valid_ratio),
        "depth_q10_mm": q10,
        "depth_q25_mm": q25,
        "depth_median_mm": q50,
        "depth_q75_mm": q75,
        "depth_q90_mm": q90,
        "depth_iqr_mm": float(depth_iqr),
        "depth_p80_span_mm": float(depth_p80_span),
        "neighbor_contact_ratio": float(contact_ratio),
        "depth_edge_ratio": float(edge_ratio),
        "mask_fill_ratio": float(mask_fill_ratio),
        "retained_pixel_count": retained_count,
        "timing_ms": _elapsed_ms(started),
    }

def _build_depth_layers(
    records: Sequence[Dict[str, Any]],
    tolerance_mm: float,
) -> List[Dict[str, Any]]:
    valid = [
        row
        for row in records
        if row.get("surface_depth_mm") is not None
        and math.isfinite(float(row["surface_depth_mm"]))
    ]
    valid.sort(
        key=lambda row: (
            float(row["surface_depth_mm"]),
            -float(row.get("depth_valid_ratio") or 0.0),
            -float(row.get("ring_confidence") or 0.0),
            int(row["ring_instance_id"]),
        )
    )
    layers: List[Dict[str, Any]] = []
    cursor = 0
    depth_rank = 1
    while cursor < len(valid):
        anchor = float(valid[cursor]["surface_depth_mm"])
        layer_rows: List[Dict[str, Any]] = []
        while cursor < len(valid):
            value = float(valid[cursor]["surface_depth_mm"])
            if value > anchor + tolerance_mm and layer_rows:
                break
            row = valid[cursor]
            row["depth_rank"] = int(depth_rank)
            depth_rank += 1
            layer_rows.append(row)
            cursor += 1
        layer_index = len(layers)
        for row in layer_rows:
            row["depth_layer_index"] = int(layer_index)
        layers.append(
            {
                "layer_index": int(layer_index),
                "anchor_depth_mm": float(anchor),
                "maximum_depth_mm": float(
                    max(float(row["surface_depth_mm"]) for row in layer_rows)
                ),
                "ring_instance_ids": [
                    int(row["ring_instance_id"]) for row in layer_rows
                ],
                "candidate_count": len(layer_rows),
                "depth_valid": True,
                "records": layer_rows,
            }
        )

    invalid = [row for row in records if row not in valid]
    if invalid:
        invalid.sort(
            key=lambda row: (
                -float(row.get("depth_valid_ratio") or 0.0),
                -float(row.get("ring_confidence") or 0.0),
                int(row["ring_instance_id"]),
            )
        )
        layer_index = len(layers)
        for row in invalid:
            row["depth_layer_index"] = int(layer_index)
            row["depth_rank"] = None
        layers.append(
            {
                "layer_index": int(layer_index),
                "anchor_depth_mm": None,
                "maximum_depth_mm": None,
                "ring_instance_ids": [
                    int(row["ring_instance_id"]) for row in invalid
                ],
                "candidate_count": len(invalid),
                "depth_valid": False,
                "records": invalid,
            }
        )
    return layers


def _scoped_geometry_config(
    geometry_config: GeometryConfig,
    allowed_ring_ids: Sequence[int],
) -> GeometryConfig:
    raw = deepcopy(geometry_config.raw)
    raw["candidate_scope"] = {
        "allowed_ring_instance_ids": [int(value) for value in allowed_ring_ids],
        "reason": "M37.5_depth_layer_active_m36_scope",
    }
    return GeometryConfig(raw)


def _scoped_m38a_geometry_config(
    geometry_config: GeometryConfig,
    allowed_ring_ids: Sequence[int],
) -> GeometryConfig:
    raw = deepcopy(geometry_config.raw)
    raw["candidate_scope"] = {
        "allowed_ring_instance_ids": [int(value) for value in allowed_ring_ids],
        "reason": "M38.1_clear_mouth_front_annulus_scope",
    }
    runtime = raw.get("_runtime")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    runtime["pose_strategy"] = "m38_1_front_annulus"
    raw["_runtime"] = runtime
    return GeometryConfig(raw)


def _deferred_side_record(
    instance: SegmentationInstance,
    *,
    attempt_rank: Optional[int],
    reason: str,
    depth_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "ring_instance_id": int(instance.instance_id),
        "ring_confidence": float(instance.confidence),
        "ring_bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "mouth_matched": False,
        "attempt_rank": int(attempt_rank) if attempt_rank is not None else None,
        "processing_status": "deferred",
        "deferred_reason": str(reason),
        "eligible": None,
        "rejection_reasons": [],
        "timing_ms": {"total_ms": 0.0},
    }
    if depth_record:
        record.update(
            {
                "surface_depth_mm": depth_record.get("surface_depth_mm"),
                "depth_layer_index": depth_record.get("depth_layer_index"),
                "depth_rank": depth_record.get("depth_rank"),
                "depth_valid_ratio": depth_record.get("depth_valid_ratio"),
            }
        )
    return record


def _fit_quality_rank(fit: Mapping[str, Any]) -> Tuple[Any, ...]:
    rejection_count = len(fit.get("rejection_reasons") or [])
    fast_reason_count = len(fit.get("fast_acceptance_reasons") or [])
    uncertainty = fit.get("pose_uncertainty") if isinstance(fit.get("pose_uncertainty"), Mapping) else {}
    bootstrap = uncertainty.get("bootstrap") if isinstance(uncertainty.get("bootstrap"), Mapping) else {}
    return (
        rejection_count,
        fast_reason_count,
        bool(uncertainty.get("ambiguous_top_hypotheses", False)),
        _safe_float(bootstrap.get("maximum_axis_spread_deg"), float("inf")),
        _safe_float(fit.get("normal_axis_p90_deg"), float("inf")),
        _safe_float(fit.get("normal_radial_p90_deg"), float("inf")),
        -_safe_float(fit.get("normal_inlier_ratio"), 0.0),
        _safe_float(fit.get("fit_score"), float("inf")),
        _safe_float(fit.get("radial_residual_p90_mm"), float("inf")),
        -_safe_float(fit.get("radial_inlier_ratio"), 0.0),
        -_safe_float(fit.get("observed_axis_span_mm"), 0.0),
        _safe_float(fit.get("surface_depth_mm"), float("inf")),
        -_safe_float(fit.get("ring_confidence"), 0.0),
    )


def _compact_fast_seed(fit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(fit.get(key))
        for key in (
            "ring_instance_id",
            "ring_confidence",
            "surface_depth_mm",
            "depth_layer_index",
            "depth_rank",
            "eligible",
            "rejection_reasons",
            "fit_score",
            "fit_model",
            "surface_counts",
            "surface_inlier_ratio",
            "surface_residual_median_mm",
            "surface_residual_p90_mm",
            "depth_gradient_axis_error_deg",
            "mouth_axis_error_deg",
            "fast_acceptance_passed",
            "fast_acceptance_reasons",
            "radial_inlier_ratio",
            "radial_residual_median_mm",
            "radial_residual_p90_mm",
            "normal_inlier_ratio",
            "normal_axis_median_deg",
            "normal_axis_p90_deg",
            "normal_radial_median_deg",
            "normal_radial_p90_deg",
            "visible_normal_span_deg",
            "pose_uncertainty",
            "observed_axis_span_mm",
            "axis_toward_camera",
            "timing_ms",
        )
    }


def _m37_candidate(fit: Mapping[str, Any]) -> Dict[str, Any]:
    crown = (
        fit.get("near_side_crown")
        if isinstance(fit.get("near_side_crown"), Mapping)
        else {}
    )
    return {
        "schema_version": "1.0",
        "message_type": "foam_ring_side_crown_grasp_candidate",
        "status": "candidate_only_not_robot_ready",
        "robot_ready": False,
        "reason": (
            "M37.6 hollow-cylinder multi-surface camera-frame grasp point only; gripper pose, hand-eye "
            "transform, reachability and final robot protocol are not enabled"
        ),
        "grasp_branch": "m37_side_ring_near_visible_crown",
        "grasp_mode": "side_ring_near_visible_crown",
        "target": {
            "ring_instance_id": fit.get("ring_instance_id"),
            "ring_confidence": fit.get("ring_confidence"),
            "attempt_rank": fit.get("attempt_rank"),
            "surface_depth_mm": fit.get("surface_depth_mm"),
            "depth_layer_index": fit.get("depth_layer_index"),
            "depth_rank": fit.get("depth_rank"),
            "fit_score": fit.get("fit_score"),
            "fit_model": fit.get("fit_model"),
            "surface_counts": deepcopy(fit.get("surface_counts") or {}),
            "surface_inlier_ratio": fit.get("surface_inlier_ratio"),
            "surface_residual_median_mm": fit.get("surface_residual_median_mm"),
            "surface_residual_p90_mm": fit.get("surface_residual_p90_mm"),
            "depth_gradient_axis_error_deg": fit.get("depth_gradient_axis_error_deg"),
            "mouth_axis_error_deg": fit.get("mouth_axis_error_deg"),
            "search_profile_used": fit.get("search_profile_used"),
            "local_accurate_refinement_used": fit.get(
                "local_accurate_refinement_used", False
            ),
            "radial_inlier_ratio": fit.get("radial_inlier_ratio"),
            "radial_residual_median_mm": fit.get(
                "radial_residual_median_mm"
            ),
            "radial_residual_p90_mm": fit.get("radial_residual_p90_mm"),
            "observed_axis_span_mm": fit.get("observed_axis_span_mm"),
            "axis_view_angle_deg": fit.get("axis_view_angle_deg"),
        },
        "grasp_point_camera_mm": crown.get("grasp_point_camera_mm"),
        "grasp_point_uv": crown.get("grasp_point_uv"),
        "axis_toward_camera": fit.get("axis_toward_camera"),
        "center_camera_mm": fit.get("center_camera_mm"),
        "near_opening_center_camera_mm": fit.get(
            "near_opening_center_camera_mm"
        ),
        "far_opening_center_camera_mm": fit.get("far_opening_center_camera_mm"),
        "near_side_crown": deepcopy(crown),
        "fit_timing_ms": deepcopy(fit.get("timing_ms") or {}),
    }


def _m38a_candidate(
    candidate: Mapping[str, Any],
    depth_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    document = deepcopy(dict(candidate))
    document["grasp_branch"] = "m38_1_clear_mouth_front_annulus_rim_pinch"
    document["grasp_mode"] = "rim_pinch"
    document["pose_source"] = "m38_1_front_annulus_depth_plane"
    target = document.get("target") if isinstance(document.get("target"), dict) else {}
    if depth_record:
        target["surface_depth_mm"] = depth_record.get("surface_depth_mm")
        target["depth_layer_index"] = depth_record.get("depth_layer_index")
        target["depth_rank"] = depth_record.get("depth_rank")
    document["target"] = target
    return document


def _m36_candidate(
    candidate: Mapping[str, Any],
    depth_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    document = deepcopy(dict(candidate))
    document["grasp_branch"] = "m36_mouth_visible_rim_pinch"
    document["grasp_mode"] = "rim_pinch"
    target = document.get("target") if isinstance(document.get("target"), dict) else {}
    if depth_record:
        target["surface_depth_mm"] = depth_record.get("surface_depth_mm")
        target["depth_layer_index"] = depth_record.get("depth_layer_index")
        target["depth_rank"] = depth_record.get("depth_rank")
    document["target"] = target
    return document


def _call_side_fit(
    side_fit_fn: Callable[..., Dict[str, Any]],
    instance: SegmentationInstance,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    template_config: SideRingTemplateConfig,
    *,
    mouth_instance: Optional[SegmentationInstance],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call M37.6-aware fitters while preserving injected legacy test doubles."""
    try:
        return side_fit_fn(
            instance, depth_mm, intrinsics, template_config,
            mouth_instance=mouth_instance, **kwargs
        )
    except TypeError as error:
        if "mouth_instance" not in str(error):
            raise
        return side_fit_fn(instance, depth_mm, intrinsics, template_config, **kwargs)


def run_hybrid_grasp(
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    raw_config: Mapping[str, Any],
    geometry_config: GeometryConfig,
    analyze_fn: Callable[..., Dict[str, Any]] = analyze_scene,
    side_fit_fn: Callable[..., Dict[str, Any]] = fit_side_ring_instance,
    associate_fn: Callable[..., Any] = _associate_ring_mouths_detailed,
) -> Dict[str, Any]:
    """Run depth-layer-first M36/M37 selection with bounded refinement."""

    hybrid_config = HybridGraspConfig.from_mapping(raw_config)
    if not hybrid_config.enabled:
        return analyze_fn(instances, depth_mm, intrinsics, geometry_config)

    total_started = time.perf_counter()
    rings = [item for item in instances if item.class_name == "foam_ring"]
    mouths = [item for item in instances if item.class_name == "ring_mouth"]

    association_started = time.perf_counter()
    matches, unmatched_rings, unmatched_mouths, association_debug = associate_fn(
        rings,
        mouths,
        geometry_config,
    )
    association_ms = _elapsed_ms(association_started)
    matched_ids = {int(ring.instance_id) for ring, _mouth, _metrics in matches}
    mouth_by_ring = {
        int(ring.instance_id): mouth for ring, mouth, _metrics in matches
    }
    ring_by_id = {int(item.instance_id): item for item in rings}

    depth_started = time.perf_counter()
    depth_records: List[Dict[str, Any]] = []
    for ring in rings:
        record = _surface_depth_record(
            ring,
            depth_mm,
            hybrid_config,
            mouth=mouth_by_ring.get(int(ring.instance_id)),
        )
        record["mouth_matched"] = int(ring.instance_id) in matched_ids
        record["branch_availability"] = (
            "m36_mouth_visible"
            if int(ring.instance_id) in matched_ids
            else "m37_side_ring"
        )
        depth_records.append(record)
    depth_preselection_ms = _elapsed_ms(depth_started)

    layer_started = time.perf_counter()
    layers = _build_depth_layers(
        depth_records,
        hybrid_config.depth_layer_tolerance_mm
        if hybrid_config.depth_layering_enabled
        else float("inf"),
    )
    depth_layer_build_ms = _elapsed_ms(layer_started)
    depth_by_id = {int(row["ring_instance_id"]): row for row in depth_records}

    template_config = SideRingTemplateConfig.from_mapping(raw_config)
    side_fits: List[Dict[str, Any]] = []
    fast_attempted_ids: set[int] = set()
    selected_side: Optional[Dict[str, Any]] = None
    selected_m38a_candidate: Optional[Dict[str, Any]] = None
    selected_m38a_scene: Optional[Dict[str, Any]] = None
    last_m38a_scene: Optional[Dict[str, Any]] = None
    selected_m36_candidate: Optional[Dict[str, Any]] = None
    selected_m36_scene: Optional[Dict[str, Any]] = None
    last_m36_scene: Optional[Dict[str, Any]] = None
    selected_branch = "none"
    selected_layer_index: Optional[int] = None
    m38a_total_ms = 0.0
    m38a_attempt_count = 0
    m36_total_ms = 0.0
    m36_attempt_count = 0
    m37_lightweight_preselection_ms = 0.0
    m37_screen_total_ms = 0.0
    m37_screen_attempt_count = 0
    m37_final_validation_ms = 0.0
    m37_final_validation_count = 0
    # Backward-compatible aliases populated from the new screen stage.
    m37_fast_total_ms = 0.0
    m37_fast_attempt_count = 0
    local_accurate_total_ms = 0.0
    accurate_refinement_count = 0
    accurate_refinement_candidate_id: Optional[int] = None
    global_side_attempt_rank = 0
    maximum_side_attempts_reached = False
    active_layer_summaries: List[Dict[str, Any]] = []

    for layer in layers:
        layer_index = int(layer["layer_index"])
        layer_ids = [int(value) for value in layer["ring_instance_ids"]]
        layer_summary: Dict[str, Any] = {
            "layer_index": layer_index,
            "anchor_depth_mm": layer.get("anchor_depth_mm"),
            "maximum_depth_mm": layer.get("maximum_depth_mm"),
            "ring_instance_ids": layer_ids,
            "m38a_candidate_ring_ids": [],
            "m36_candidate_ring_ids": [],
            "m37_candidate_ring_ids": [],
            "m38a_attempted": False,
            "m38a_candidate_found": False,
            "m36_attempted": False,
            "m36_candidate_found": False,
            "m37_fast_attempted_ids": [],
            "m37_fast_accepted_id": None,
            "local_accurate_refined_id": None,
            "selected_branch": "none",
        }

        mouth_visible_ids = [value for value in layer_ids if value in matched_ids]
        layer_summary["m38a_candidate_ring_ids"] = list(mouth_visible_ids)
        layer_summary["m36_candidate_ring_ids"] = list(mouth_visible_ids)
        if mouth_visible_ids and hybrid_config.prefer_mouth_visible:
            if hybrid_config.m38_branch_a_enabled:
                m38a_attempt_count += 1
                layer_summary["m38a_attempted"] = True
                m38a_started = time.perf_counter()
                m38a_scene = analyze_fn(
                    instances,
                    depth_mm,
                    intrinsics,
                    _scoped_m38a_geometry_config(
                        geometry_config,
                        mouth_visible_ids,
                    ),
                )
                m38a_elapsed = _elapsed_ms(m38a_started)
                m38a_total_ms += m38a_elapsed
                last_m38a_scene = m38a_scene
                layer_summary["m38a_ms"] = float(m38a_elapsed)
                original_m38a = (
                    m38a_scene.get("robot_candidate")
                    if isinstance(m38a_scene.get("robot_candidate"), Mapping)
                    else None
                )
                if original_m38a is not None:
                    ring_id = int(
                        (original_m38a.get("target") or {}).get("ring_instance_id")
                    )
                    selected_m38a_candidate = _m38a_candidate(
                        original_m38a,
                        depth_by_id.get(ring_id),
                    )
                    selected_m38a_scene = m38a_scene
                    selected_branch = "m38_1_clear_mouth_front_annulus_rim_pinch"
                    selected_layer_index = layer_index
                    layer_summary["m38a_candidate_found"] = True
                    layer_summary["selected_branch"] = selected_branch
                    active_layer_summaries.append(layer_summary)
                    break

            if (
                selected_m38a_candidate is None
                and (
                    not hybrid_config.m38_branch_a_enabled
                    or hybrid_config.m38_branch_a_fallback_to_m36
                )
            ):
                m36_attempt_count += 1
                layer_summary["m36_attempted"] = True
                m36_started = time.perf_counter()
                m36_scene = analyze_fn(
                    instances,
                    depth_mm,
                    intrinsics,
                    _scoped_geometry_config(geometry_config, mouth_visible_ids),
                )
                m36_elapsed = _elapsed_ms(m36_started)
                m36_total_ms += m36_elapsed
                last_m36_scene = m36_scene
                layer_summary["m36_ms"] = float(m36_elapsed)
                original = (
                    m36_scene.get("robot_candidate")
                    if isinstance(m36_scene.get("robot_candidate"), Mapping)
                    else None
                )
                if original is not None:
                    ring_id = int((original.get("target") or {}).get("ring_instance_id"))
                    selected_m36_candidate = _m36_candidate(
                        original,
                        depth_by_id.get(ring_id),
                    )
                    selected_m36_scene = m36_scene
                    selected_branch = "m36_mouth_visible_rim_pinch"
                    selected_layer_index = layer_index
                    layer_summary["m36_candidate_found"] = True
                    layer_summary["selected_branch"] = selected_branch
                    active_layer_summaries.append(layer_summary)
                    break

        if not hybrid_config.side_ring_fallback_enabled:
            active_layer_summaries.append(layer_summary)
            continue

        side_candidates = [
            ring_by_id[value]
            for value in layer_ids
            if value in ring_by_id
            and (
                hybrid_config.multi_surface_include_m36_rejected
                or not hybrid_config.side_ring_only_unmatched
                or value not in matched_ids
            )
        ]

        # M37.5.1 stage 1: millisecond-level candidate ranking. No cylinder
        # search, Top-K ambiguity or bootstrap is executed here.
        lightweight_started = time.perf_counter()
        lightweight_records: List[Dict[str, Any]] = []
        for instance in side_candidates:
            instance_id = int(instance.instance_id)
            record = _lightweight_side_candidate_record(
                instance,
                rings,
                depth_mm,
                depth_by_id[instance_id],
                hybrid_config,
            )
            lightweight_records.append(record)
        lightweight_records.sort(
            key=lambda row: (
                not bool(row.get("eligible_for_screen", False)),
                # Physical top-surface order remains primary even inside one
                # 30 mm layer; lightweight quality only breaks near ties.
                _safe_float(row.get("surface_depth_mm"), float("inf")),
                _safe_float(row.get("score"), float("inf")),
                -_safe_float(row.get("ring_confidence"), 0.0),
                _safe_int(row.get("ring_instance_id"), 10**9),
            )
        )
        for rank, record in enumerate(lightweight_records, start=1):
            record["preselection_rank"] = int(rank)
        lightweight_elapsed = _elapsed_ms(lightweight_started)
        m37_lightweight_preselection_ms += lightweight_elapsed
        lightweight_by_id = {
            int(row["ring_instance_id"]): row for row in lightweight_records
        }
        if hybrid_config.lightweight_preselection_enabled:
            screen_candidates = [
                ring_by_id[int(row["ring_instance_id"])]
                for row in lightweight_records
                if bool(row.get("eligible_for_screen", False))
            ][: hybrid_config.lightweight_maximum_candidates_per_layer]
        else:
            screen_candidates = list(side_candidates)
            screen_candidates.sort(
                key=lambda instance: (
                    _safe_float(
                        depth_by_id[int(instance.instance_id)].get("surface_depth_mm"),
                        float("inf"),
                    ),
                    -float(instance.confidence),
                    int(instance.instance_id),
                )
            )
        layer_summary["m37_candidate_ring_ids"] = [
            int(item.instance_id) for item in side_candidates
        ]
        layer_summary["m37_lightweight_preselection_ms"] = float(lightweight_elapsed)
        layer_summary["m37_lightweight_candidates"] = lightweight_records
        layer_summary["m37_screen_candidate_ring_ids"] = [
            int(item.instance_id) for item in screen_candidates
        ]
        layer_summary["m37_screen_attempted_ids"] = []
        layer_summary["m37_final_validated_ids"] = []
        preliminary_fits: List[Dict[str, Any]] = []
        final_failed_seeds: List[Dict[str, Any]] = []

        # M37.5.1 stage 2/3: cheap preliminary pose, then delayed full safety
        # validation only for the current candidate. fit_score ranks quality but
        # no longer vetoes an otherwise pose-safe result.
        for instance in screen_candidates:
            if (
                hybrid_config.maximum_side_ring_attempts > 0
                and m37_screen_attempt_count >= hybrid_config.maximum_side_ring_attempts
            ):
                maximum_side_attempts_reached = True
                break
            global_side_attempt_rank += 1
            screen_started = time.perf_counter()
            screen_fit = _call_side_fit(
                side_fit_fn, instance, depth_mm, intrinsics, template_config,
                mouth_instance=mouth_by_ring.get(int(instance.instance_id)),
                mouth_matched=int(instance.instance_id) in matched_ids,
                search_profile="screen",
                exclusion_mask=_other_ring_exclusion_mask(instance, rings),
            )
            screen_elapsed = _elapsed_ms(screen_started)
            m37_screen_total_ms += screen_elapsed
            m37_fast_total_ms += screen_elapsed
            m37_screen_attempt_count += 1
            m37_fast_attempt_count += 1
            instance_id = int(instance.instance_id)
            fast_attempted_ids.add(instance_id)
            screen_fit["attempt_rank"] = int(global_side_attempt_rank)
            screen_fit["processing_status"] = "screen_evaluated"
            screen_fit["surface_depth_mm"] = depth_by_id[instance_id].get(
                "surface_depth_mm"
            )
            screen_fit["depth_layer_index"] = depth_by_id[instance_id].get(
                "depth_layer_index"
            )
            screen_fit["depth_rank"] = depth_by_id[instance_id].get("depth_rank")
            screen_fit["depth_valid_ratio"] = depth_by_id[instance_id].get(
                "depth_valid_ratio"
            )
            screen_fit["lightweight_preselection"] = deepcopy(
                lightweight_by_id.get(instance_id) or {}
            )
            screen_fit["screen_wall_ms"] = float(screen_elapsed)
            screen_fit["fast_wall_ms"] = float(screen_elapsed)
            side_fits.append(screen_fit)
            preliminary_fits.append(screen_fit)
            layer_summary["m37_screen_attempted_ids"].append(instance_id)
            layer_summary["m37_fast_attempted_ids"].append(instance_id)

            if not bool(screen_fit.get("preliminary_pose_safe", screen_fit.get("eligible", False))):
                continue

            if not hybrid_config.delayed_final_validation_enabled:
                selected_side = screen_fit
                selected_branch = "m37_side_ring_near_visible_crown"
                selected_layer_index = layer_index
                layer_summary["m37_fast_accepted_id"] = instance_id
                layer_summary["selected_branch"] = selected_branch
                break

            final_started = time.perf_counter()
            final_fit = _call_side_fit(
                side_fit_fn, instance, depth_mm, intrinsics, template_config,
                mouth_instance=mouth_by_ring.get(int(instance.instance_id)),
                mouth_matched=int(instance.instance_id) in matched_ids,
                search_profile="final_verify",
                initial_axis=np.asarray(
                    screen_fit["axis_toward_camera"], dtype=np.float64
                ),
                exclusion_mask=_other_ring_exclusion_mask(instance, rings),
            )
            final_elapsed = _elapsed_ms(final_started)
            m37_final_validation_ms += final_elapsed
            m37_final_validation_count += 1
            final_fit["attempt_rank"] = screen_fit.get("attempt_rank")
            final_fit["processing_status"] = "final_validated"
            final_fit["surface_depth_mm"] = screen_fit.get("surface_depth_mm")
            final_fit["depth_layer_index"] = screen_fit.get("depth_layer_index")
            final_fit["depth_rank"] = screen_fit.get("depth_rank")
            final_fit["depth_valid_ratio"] = screen_fit.get("depth_valid_ratio")
            final_fit["lightweight_preselection"] = deepcopy(
                screen_fit.get("lightweight_preselection") or {}
            )
            final_fit["screen_seed"] = _compact_fast_seed(screen_fit)
            final_fit["screen_wall_ms"] = float(screen_elapsed)
            final_fit["final_validation_wall_ms"] = float(final_elapsed)
            for index, existing in enumerate(side_fits):
                if existing is screen_fit:
                    side_fits[index] = final_fit
                    break
            layer_summary["m37_final_validated_ids"].append(instance_id)

            if bool(final_fit.get("final_pose_safe", final_fit.get("eligible", False))):
                selected_side = final_fit
                selected_branch = "m37_side_ring_near_visible_crown"
                selected_layer_index = layer_index
                layer_summary["m37_fast_accepted_id"] = instance_id
                layer_summary["selected_branch"] = selected_branch
                break
            final_failed_seeds.append(final_fit)

        if selected_side is not None:
            active_layer_summaries.append(layer_summary)
            break

        # M37.5.1 stage 4: only when no candidate passes delayed final safety,
        # allow one bounded warm-start local refinement for the best preliminary
        # axis. This path never re-runs a global direction search.
        if (
            preliminary_fits
            and accurate_refinement_count
            < hybrid_config.maximum_accurate_refinements_per_trigger
        ):
            refinable = [
                fit
                for fit in preliminary_fits
                if fit.get("axis_toward_camera") is not None
            ]
            if refinable:
                best_screen = min(refinable, key=_fit_quality_rank)
                best_id = int(best_screen["ring_instance_id"])
                best_instance = ring_by_id[best_id]
                local_started = time.perf_counter()
                refined = _call_side_fit(
                    side_fit_fn, best_instance, depth_mm, intrinsics, template_config,
                    mouth_instance=mouth_by_ring.get(best_id),
                    mouth_matched=best_id in matched_ids,
                    search_profile="local_accurate",
                    initial_axis=np.asarray(
                        best_screen["axis_toward_camera"], dtype=np.float64
                    ),
                    exclusion_mask=_other_ring_exclusion_mask(best_instance, rings),
                )
                local_elapsed = _elapsed_ms(local_started)
                local_accurate_total_ms += local_elapsed
                accurate_refinement_count += 1
                accurate_refinement_candidate_id = best_id
                refined["attempt_rank"] = best_screen.get("attempt_rank")
                refined["processing_status"] = "local_accurate_refined"
                refined["surface_depth_mm"] = best_screen.get("surface_depth_mm")
                refined["depth_layer_index"] = best_screen.get(
                    "depth_layer_index"
                )
                refined["depth_rank"] = best_screen.get("depth_rank")
                refined["depth_valid_ratio"] = best_screen.get(
                    "depth_valid_ratio"
                )
                refined["lightweight_preselection"] = deepcopy(
                    best_screen.get("lightweight_preselection") or {}
                )
                refined["local_accurate_refinement_used"] = True
                refined["local_accurate_wall_ms"] = float(local_elapsed)
                refined["screen_seed"] = _compact_fast_seed(best_screen)
                replaced = False
                for index, existing in enumerate(side_fits):
                    if int(existing.get("ring_instance_id", -1)) == best_id:
                        side_fits[index] = refined
                        replaced = True
                        break
                if not replaced:
                    side_fits.append(refined)
                layer_summary["local_accurate_refined_id"] = best_id
                if bool(refined.get("eligible", False)):
                    selected_side = refined
                    selected_branch = "m37_side_ring_near_visible_crown"
                    selected_layer_index = layer_index
                    layer_summary["selected_branch"] = selected_branch

        active_layer_summaries.append(layer_summary)
        if selected_side is not None or maximum_side_attempts_reached:
            break

    # Reuse the selected branch scene so its pose diagnostics and instance
    # rejection reasons remain visible.  The cheap empty M36 scope is only
    # needed when neither mouth-visible branch ran.
    m36_base_scene_ms = 0.0
    if selected_m38a_scene is not None:
        result_scene = selected_m38a_scene
    elif selected_m36_scene is not None:
        result_scene = selected_m36_scene
    elif last_m36_scene is not None:
        result_scene = last_m36_scene
    elif last_m38a_scene is not None:
        result_scene = last_m38a_scene
    else:
        base_started = time.perf_counter()
        result_scene = analyze_fn(
            instances,
            depth_mm,
            intrinsics,
            _scoped_geometry_config(geometry_config, []),
        )
        m36_base_scene_ms = _elapsed_ms(base_started)
        m36_total_ms += m36_base_scene_ms
    if not isinstance(result_scene, dict):
        raise ValueError("mouth-visible analyze_scene must return a dict")

    # Explicitly mark every unattempted side candidate as deferred.
    side_candidate_ids = {
        int(ring.instance_id)
        for ring in rings
        if (
            not hybrid_config.side_ring_only_unmatched
            or int(ring.instance_id) not in matched_ids
        )
    }
    deferred_reason = (
        "after_depth_layer_first_valid"
        if selected_branch != "none"
        else (
            "maximum_side_ring_attempts_reached"
            if maximum_side_attempts_reached
            else "not_reached_before_search_exhaustion"
        )
    )
    for instance_id in sorted(side_candidate_ids - fast_attempted_ids):
        side_fits.append(
            _deferred_side_record(
                ring_by_id[instance_id],
                attempt_rank=None,
                reason=deferred_reason,
                depth_record=depth_by_id.get(instance_id),
            )
        )

    selected_candidate: Optional[Dict[str, Any]]
    original_m38a_candidate = (
        selected_m38a_scene.get("robot_candidate")
        if selected_m38a_scene is not None
        and isinstance(selected_m38a_scene.get("robot_candidate"), Mapping)
        else None
    )
    original_m36_candidate = (
        selected_m36_scene.get("robot_candidate")
        if selected_m36_scene is not None
        and isinstance(selected_m36_scene.get("robot_candidate"), Mapping)
        else None
    )
    if selected_m38a_candidate is not None:
        selected_candidate = selected_m38a_candidate
    elif selected_m36_candidate is not None:
        selected_candidate = selected_m36_candidate
    elif selected_side is not None:
        selected_candidate = _m37_candidate(selected_side)
    else:
        selected_candidate = None

    evaluated_side = [
        fit
        for fit in side_fits
        if fit.get("processing_status")
        in {"screen_evaluated", "final_validated", "local_accurate_refined", "fast_evaluated"}
    ]
    deferred_side = [
        fit for fit in side_fits if fit.get("processing_status") == "deferred"
    ]
    selected_side_id = (
        int(selected_side.get("ring_instance_id"))
        if selected_side is not None
        else None
    )
    selected_ring_id = None
    if selected_candidate is not None:
        selected_ring_id_raw = (selected_candidate.get("target") or {}).get(
            "ring_instance_id"
        )
        if selected_ring_id_raw is not None:
            selected_ring_id = int(selected_ring_id_raw)
    selected_depth = depth_by_id.get(selected_ring_id) if selected_ring_id is not None else None
    total_ms = _elapsed_ms(total_started)

    result = dict(result_scene)
    result["rings_detected"] = len(rings)
    result["mouths_detected"] = len(mouths)
    result["global_matched_pairs"] = len(matches)
    result["association_debug"] = association_debug
    result["unmatched_ring_ids"] = [int(item.instance_id) for item in unmatched_rings]
    result["unmatched_mouth_ids"] = [int(item.instance_id) for item in unmatched_mouths]
    result["m38_1_eligible_count"] = (
        selected_m38a_scene.get("eligible_count")
        if selected_m38a_scene is not None
        else (last_m38a_scene.get("eligible_count") if last_m38a_scene is not None else 0)
    )
    result["m36_eligible_count"] = (
        selected_m36_scene.get("eligible_count")
        if selected_m36_scene is not None
        else (last_m36_scene.get("eligible_count") if last_m36_scene is not None else 0)
    )
    if selected_side is not None:
        result["eligible_count"] = 1
        result["selected_ring_instance_id"] = selected_side_id
        result["selected_clock_hour"] = None
        result["selected_clock_angle_deg_cw_from_12"] = None
        result["selected_clock_search_batch"] = None
    result["m38_1_robot_candidate"] = deepcopy(original_m38a_candidate)
    result["m36_robot_candidate"] = deepcopy(original_m36_candidate)
    result["robot_candidate"] = selected_candidate
    result["selected_grasp_branch"] = selected_branch
    result["depth_layering"] = {
        "enabled": bool(hybrid_config.depth_layering_enabled),
        "ordering_rule": (
            "depth_layer_ascending_then_M38.1_annulus_then_m36_then_M37.6_multisurface"
        ),
        "surface_depth_statistic": f"p{hybrid_config.surface_depth_percentile:g}",
        "layer_tolerance_mm": float(hybrid_config.depth_layer_tolerance_mm),
        "candidate_count": len(depth_records),
        "valid_depth_candidate_count": sum(
            1 for row in depth_records if row.get("surface_depth_mm") is not None
        ),
        "layers": [
            {key: deepcopy(value) for key, value in layer.items() if key != "records"}
            for layer in layers
        ],
        "candidates": depth_records,
        "processed_layers": active_layer_summaries,
        "selected_layer_index": selected_layer_index,
        "selected_surface_depth_mm": (
            selected_depth.get("surface_depth_mm") if selected_depth else None
        ),
        "selected_depth_rank": selected_depth.get("depth_rank") if selected_depth else None,
    }
    result["hybrid_grasp"] = {
        "enabled": True,
        "policy_version": "M38.1",
        "branch_priority": [
            "nearest_depth_layer",
            "same_layer_M38.1_clear_mouth_front_annulus_rim_pinch",
            "same_layer_m36_legacy_fallback",
            "same_layer_m37_lightweight_preselection",
            "preliminary_pose_screen",
            "delayed_final_pose_validation",
            "single_warm_start_local_accurate_refinement_if_needed",
            "next_depth_layer",
        ],
        "selected_branch": selected_branch,
        "selected_depth_layer_index": selected_layer_index,
        "fallback_triggered": bool(
            selected_branch in {
                "m36_mouth_visible_rim_pinch",
                "m37_side_ring_near_visible_crown",
            }
        ),
        "m38_1_candidate_found": bool(selected_m38a_candidate is not None),
        "m36_candidate_found": bool(selected_m36_candidate is not None),
        "m37_candidate_found": bool(selected_side is not None),
        "target_found": bool(selected_candidate is not None),
        "timing_ms": {
            "association_prepass_ms": float(association_ms),
            "depth_preselection_ms": float(depth_preselection_ms),
            "depth_layer_build_ms": float(depth_layer_build_ms),
            "m38_1_branch_a_ms": float(m38a_total_ms),
            "m36_branch_ms": float(m36_total_ms),
            "m36_base_scene_ms": float(m36_base_scene_ms),
            "m37_lightweight_preselection_ms": float(m37_lightweight_preselection_ms),
            "m37_screen_total_ms": float(m37_screen_total_ms),
            "m37_final_validation_ms": float(m37_final_validation_ms),
            "m37_fast_total_ms": float(m37_fast_total_ms),
            "m37_local_accurate_ms": float(local_accurate_total_ms),
            "m37_evaluated_instance_total_ms": float(
                sum(
                    _safe_float((fit.get("timing_ms") or {}).get("total_ms"), 0.0)
                    for fit in evaluated_side
                )
            ),
            "total_ms": float(total_ms),
        },
    }
    m38a_diagnostic_scene = selected_m38a_scene or last_m38a_scene
    m38a_pair_results: List[Dict[str, Any]] = []
    if isinstance(m38a_diagnostic_scene, Mapping):
        for item in m38a_diagnostic_scene.get("instances") or []:
            if not isinstance(item, Mapping):
                continue
            annulus = item.get("m38_branch_a")
            if not isinstance(annulus, Mapping):
                continue
            plane = item.get("plane") if isinstance(item.get("plane"), Mapping) else {}
            m38a_pair_results.append({
                "ring_instance_id": item.get("ring_instance_id"),
                "mouth_instance_id": item.get("mouth_instance_id"),
                "eligible": bool(item.get("eligible", False)),
                "tilt_deg": item.get("tilt_deg"),
                "opening_clear": bool(annulus.get("opening_clear", False)),
                "annulus_point_count": annulus.get("annulus_point_count"),
                "annulus_depth_valid_ratio": annulus.get("annulus_depth_valid_ratio"),
                "angular_coverage_deg": annulus.get("angular_coverage_deg"),
                "inlier_angular_coverage_deg": annulus.get("inlier_angular_coverage_deg"),
                "plane_inlier_ratio": plane.get("inlier_ratio"),
                "plane_residual_p95_mm": plane.get("residual_p95_mm"),
                "rejection_reasons": list(item.get("rejection_reasons") or []),
                "warnings": list(item.get("warnings") or []),
            })
    result["m38_1_branch_a"] = {
        "enabled": bool(hybrid_config.m38_branch_a_enabled),
        "pose_source": "direct_3d_front_annulus_plane",
        "fallback_to_m36": bool(hybrid_config.m38_branch_a_fallback_to_m36),
        "attempt_count": int(m38a_attempt_count),
        "candidate_found": bool(selected_m38a_candidate is not None),
        "selected_ring_instance_id": (
            int((selected_m38a_candidate.get("target") or {}).get("ring_instance_id"))
            if selected_m38a_candidate is not None
            else None
        ),
        "pair_results": m38a_pair_results,
        "timing_ms": float(m38a_total_ms),
        "legacy_m36_retained": True,
        "m37_6_retained": True,
    }
    result["side_ring_branch"] = {
        "enabled": bool(hybrid_config.side_ring_fallback_enabled),
        "executed": bool(m37_fast_attempt_count > 0),
        "candidate_order_rule": "depth_layer_then_surface_depth_ascending",
        "only_unmatched_m36_rings": bool(hybrid_config.side_ring_only_unmatched),
        "search_policy": "lightweight_rank_then_screen_then_delayed_final_validation",
        "candidate_count": int(len(evaluated_side) + len(deferred_side)),
        "evaluated_count": int(len(evaluated_side)),
        "lightweight_preselection_enabled": bool(hybrid_config.lightweight_preselection_enabled),
        "lightweight_maximum_candidates_per_layer": int(
            hybrid_config.lightweight_maximum_candidates_per_layer
        ),
        "screen_attempt_count": int(m37_screen_attempt_count),
        "final_validation_count": int(m37_final_validation_count),
        "fast_attempt_count": int(m37_fast_attempt_count),
        "deferred_count": int(len(deferred_side)),
        "selected_ring_instance_id": selected_side_id,
        "accurate_refinement_count": int(accurate_refinement_count),
        "maximum_accurate_refinements_per_trigger": int(
            hybrid_config.maximum_accurate_refinements_per_trigger
        ),
        "accurate_refinement_candidate_id": accurate_refinement_candidate_id,
        "global_accurate_search_used": False,
        "fits": side_fits,
    }
    m36_pose_conflict_rejections = 0
    for item in result.get("instances") or []:
        reasons = item.get("rejection_reasons") or []
        if any(
            reason in {
                "depth_plane_ellipse_pose_conflict",
                "ellipse_pose_has_insufficient_depth_support",
                "ellipse_stabilized_pose_residual_too_high",
                "pose_conflict_fallback_to_m37",
            }
            for reason in reasons
        ):
            m36_pose_conflict_rejections += 1
    m37_uncertainty_rejections = sum(
        1
        for fit in evaluated_side
        if any(
            reason in {
                "axis_hypotheses_ambiguous",
                "axis_bootstrap_unstable",
                "axis_disagrees_with_surface_normal_seed",
                "surface_normal_inlier_ratio_too_low",
                "surface_normals_not_perpendicular_to_axis",
                "surface_normal_axis_p90_too_high",
                "surface_normals_not_radial",
                "surface_normal_radial_p90_too_high",
                "visible_cylindrical_normal_span_too_small",
            }
            for reason in (fit.get("rejection_reasons") or [])
        )
    )
    m36_pose_conflict_handoffs = sum(
        1
        for item in result.get("instances") or []
        if "pose_conflict_fallback_to_m37" in (item.get("rejection_reasons") or [])
    )
    result["m37_5_1_pose_safety"] = {
        "normal_constrained_axis_enabled": bool(template_config.normal_constrained_enabled),
        "neighbor_surface_exclusion_enabled": True,
        "m36_pose_conflict_rejection_count": int(m36_pose_conflict_rejections),
        "m36_pose_conflict_handoff_count": int(m36_pose_conflict_handoffs),
        "m37_uncertainty_rejection_count": int(m37_uncertainty_rejections),
        "selected_branch": selected_branch,
        "selected_ring_instance_id": selected_ring_id,
        "selected_pose_uncertainty": (
            deepcopy(selected_side.get("pose_uncertainty"))
            if selected_side is not None
            else None
        ),
    }
    # Backward-compatible M37.5 alias.
    result["m37_5_pose_safety"] = deepcopy(result["m37_5_1_pose_safety"])
    result["m37_5_timing"] = {
        "depth_preselection_ms": float(depth_preselection_ms),
        "depth_layer_build_ms": float(depth_layer_build_ms),
        "m36_attempt_count": int(m36_attempt_count),
        "m36_total_ms": float(m36_total_ms),
        "m37_lightweight_preselection_ms": float(m37_lightweight_preselection_ms),
        "m37_screen_attempt_count": int(m37_screen_attempt_count),
        "m37_screen_total_ms": float(m37_screen_total_ms),
        "m37_final_validation_count": int(m37_final_validation_count),
        "m37_final_validation_ms": float(m37_final_validation_ms),
        "m37_fast_attempt_count": int(m37_fast_attempt_count),
        "m37_fast_total_ms": float(m37_fast_total_ms),
        "m37_local_accurate_ms": float(local_accurate_total_ms),
        "pose_safety_rejection_count": int(
            m36_pose_conflict_rejections + m37_uncertainty_rejections
        ),
        "total_ms": float(total_ms),
    }

    result["m37_5_1_timing"] = deepcopy(result["m37_5_timing"])

    # Backward-compatible M37.4 timing alias retained for existing dashboards.
    result["m37_4_timing"] = {
        "depth_preselection_ms": float(depth_preselection_ms),
        "depth_layer_build_ms": float(depth_layer_build_ms),
        "m36_attempt_count": int(m36_attempt_count),
        "m36_total_ms": float(m36_total_ms),
        "m36_base_scene_ms": float(m36_base_scene_ms),
        "m37_fast_attempt_count": int(m37_fast_attempt_count),
        "m37_fast_total_ms": float(m37_fast_total_ms),
        "accurate_refine_used": bool(accurate_refinement_count > 0),
        "accurate_refine_candidate_id": accurate_refinement_candidate_id,
        "accurate_local_refine_ms": float(local_accurate_total_ms),
        "selected_surface_depth_mm": (
            selected_depth.get("surface_depth_mm") if selected_depth else None
        ),
        "selected_depth_layer_index": selected_layer_index,
        "selected_depth_rank": selected_depth.get("depth_rank") if selected_depth else None,
        "total_ms": float(total_ms),
    }
    return result
