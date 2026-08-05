"""M38.3 depth-evidence partial-opening constrained-cylinder recovery.

M38.2 required a segmented ``ring_mouth``.  Real RKNN results showed that the
partially visible opening of a side/half-side ring is often not emitted as a
``ring_mouth`` at all; meanwhile a nearly concentric clear mouth rejected by
M38.1 could be incorrectly routed into branch B.  M38.3 therefore supports two
opening-evidence sources:

* an explicitly segmented, genuinely off-centre partial mouth;
* a depth-inferred aperture inside an unmatched ``foam_ring`` mask.

The cylinder axis is no longer searched freely in 3-D.  Its image projection is
constrained to point from the ring body toward the observed aperture, and only
the camera-facing view component is sampled.  This removes the orthogonal-axis
minimum observed in M38.2 and reduces the axis candidates substantially.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .segmentation import SegmentationInstance
from .partial_opening_cylinder import (
    _basis_perpendicular,
    _bbox_from_mask,
    _deproject_mask,
    _depth_edge_mask,
    _dilate,
    _erode,
    _evaluate_axis,
    _kernel,
    _organized_normals,
    _project_points,
    _retain_components,
    _safe_float,
    _safe_int,
    _unit,
)

_EPS = 1e-9


def _surface_depth_mode(depth_mm: np.ndarray, mask: np.ndarray) -> Optional[float]:
    values = depth_mm[mask.astype(bool) & (depth_mm > 0)].astype(np.float64)
    if values.size < 40:
        return None
    upper = float(np.percentile(values, 70.0))
    lower_values = values[values <= upper]
    if lower_values.size < 20:
        lower_values = values
    low = float(np.min(lower_values))
    high = float(np.max(lower_values))
    if high - low < 2.0:
        return float(np.median(lower_values))
    bin_width = 2.0
    edges = np.arange(low - 0.5 * bin_width, high + 1.5 * bin_width, bin_width)
    histogram, edges = np.histogram(lower_values, bins=edges)
    index = int(np.argmax(histogram))
    return float(0.5 * (edges[index] + edges[index + 1]))


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), _kernel(1), iterations=1).astype(bool)
    return mask.astype(bool) & ~eroded


def infer_depth_partial_opening(
    ring: SegmentationInstance,
    depth_mm: np.ndarray,
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Infer a partial aperture from a deep connected component inside a ring.

    This is an evidence detector only.  A returned component still has to pass
    the constrained cylinder, rim-anchor and projected-opening checks.
    """

    started = time.perf_counter()
    section = raw_config.get("m38_branch_b") or {}
    if not isinstance(section, Mapping):
        section = {}
    depth_cfg = raw_config.get("depth") or {}
    if not isinstance(depth_cfg, Mapping):
        depth_cfg = {}

    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    eroded = _erode(ring.mask, _safe_int(section.get("inferred_opening_ring_erode_px"), 1))
    eroded &= (depth_mm >= minimum_depth) & (depth_mm <= maximum_depth)
    surface_depth = _surface_depth_mode(depth_mm, eroded)
    if surface_depth is None:
        return {
            "eligible": False,
            "rejection_reasons": ["m383_depth_opening_surface_depth_unavailable"],
            "diagnostics": {"opening_source": "depth_inferred", "surface_depth_mm": None},
            "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
            "_debug": {},
        }

    minimum_gap = _safe_float(section.get("inferred_opening_minimum_depth_gap_mm"), 35.0)
    deep_mask = eroded & (depth_mm.astype(np.float64) >= surface_depth + minimum_gap)
    deep_u8 = deep_mask.astype(np.uint8)
    deep_u8 = cv2.morphologyEx(deep_u8, cv2.MORPH_OPEN, _kernel(1), iterations=1)
    deep_u8 = cv2.morphologyEx(deep_u8, cv2.MORPH_CLOSE, _kernel(1), iterations=1)
    deep_mask = deep_u8.astype(bool)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        deep_mask.astype(np.uint8), connectivity=8
    )
    ring_area = max(1, int(ring.area_px))
    x1, y1, x2, y2 = ring.bbox_xyxy
    diagonal = max(1.0, math.hypot(float(x2 - x1), float(y2 - y1)))
    ring_center = np.asarray(ring.centroid_uv, dtype=np.float64)
    distance_inside = cv2.distanceTransform(ring.mask.astype(np.uint8), cv2.DIST_L2, 5)
    ring_boundary = _mask_boundary(ring.mask)
    dilated_boundary = _dilate(ring_boundary, 1)

    min_area_ratio = _safe_float(section.get("inferred_opening_minimum_area_ratio"), 0.04)
    max_area_ratio = _safe_float(section.get("inferred_opening_maximum_area_ratio"), 0.36)
    min_offset_ratio = _safe_float(section.get("inferred_opening_minimum_center_offset_ratio"), 0.12)
    max_offset_ratio = _safe_float(section.get("inferred_opening_maximum_center_offset_ratio"), 0.68)
    max_boundary_contact = _safe_float(section.get("inferred_opening_maximum_boundary_contact_ratio"), 0.68)
    min_inside_distance = _safe_float(section.get("inferred_opening_minimum_inside_distance_px"), 5.0)
    min_rim_points = _safe_int(section.get("inferred_opening_minimum_rim_support_px"), 16)

    candidates: List[Dict[str, Any]] = []
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = float(area) / float(ring_area)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        component = labels == label
        center = np.asarray(centroids[label], dtype=np.float64)
        offset_px = float(np.linalg.norm(center - ring_center))
        offset_ratio = offset_px / diagonal
        if not (min_offset_ratio <= offset_ratio <= max_offset_ratio):
            continue
        max_inside = float(np.max(distance_inside[component])) if np.any(component) else 0.0
        if max_inside < min_inside_distance:
            continue
        component_boundary = _mask_boundary(component)
        boundary_contact = float(np.count_nonzero(component_boundary & dilated_boundary)) / float(
            max(1, np.count_nonzero(component_boundary))
        )
        if boundary_contact > max_boundary_contact:
            continue
        values = depth_mm[component].astype(np.float64)
        median_depth = float(np.median(values)) if values.size else 0.0
        depth_gap = median_depth - surface_depth
        if depth_gap < minimum_gap:
            continue
        rim_support = (
            _dilate(component, _safe_int(section.get("inferred_opening_rim_dilate_px"), 5))
            & ring.mask
            & ~component
            & (depth_mm >= surface_depth - _safe_float(section.get("side_depth_front_tolerance_mm"), 18.0))
            & (depth_mm <= surface_depth + _safe_float(section.get("rim_support_depth_back_tolerance_mm"), 50.0))
        )
        rim_count = int(np.count_nonzero(rim_support))
        if rim_count < min_rim_points:
            continue
        score = (
            2.0 * min(1.0, depth_gap / 80.0)
            + 1.5 * min(1.0, area_ratio / 0.18)
            + 1.0 * min(1.0, max_inside / 20.0)
            + 0.7 * min(1.0, offset_ratio / 0.30)
            - 0.8 * boundary_contact
        )
        candidates.append({
            "score": float(score),
            "mask": component,
            "rim_support_mask": rim_support,
            "area_px": area,
            "area_ratio": area_ratio,
            "centroid_uv": center,
            "center_offset_px": offset_px,
            "center_offset_ratio": offset_ratio,
            "surface_depth_mm": float(surface_depth),
            "median_depth_mm": median_depth,
            "depth_gap_mm": float(depth_gap),
            "maximum_inside_distance_px": max_inside,
            "boundary_contact_ratio": boundary_contact,
            "rim_support_pixel_count": rim_count,
        })

    if not candidates:
        return {
            "eligible": False,
            "rejection_reasons": ["m383_depth_partial_opening_not_found"],
            "diagnostics": {
                "opening_source": "depth_inferred",
                "surface_depth_mm": float(surface_depth),
                "component_count": max(0, int(count) - 1),
            },
            "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
            "_debug": {"deep_candidate_mask": deep_mask},
        }

    selected = max(candidates, key=lambda row: float(row["score"]))
    opening_mask = np.asarray(selected["mask"], dtype=bool)
    ys, xs = np.nonzero(opening_mask)
    mouth = SegmentationInstance(
        instance_id=-100000 - int(ring.instance_id),
        class_id=1,
        class_name="ring_mouth",
        confidence=min(0.99, max(0.05, float(selected["score"]) / 5.0)),
        mask=opening_mask,
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    )
    diagnostics = {
        "opening_source": "depth_inferred",
        "surface_depth_mm": float(selected["surface_depth_mm"]),
        "opening_depth_median_mm": float(selected["median_depth_mm"]),
        "opening_depth_gap_mm": float(selected["depth_gap_mm"]),
        "opening_area_px": int(selected["area_px"]),
        "opening_area_ratio": float(selected["area_ratio"]),
        "opening_center_offset_px": float(selected["center_offset_px"]),
        "opening_center_offset_ratio": float(selected["center_offset_ratio"]),
        "opening_maximum_inside_distance_px": float(selected["maximum_inside_distance_px"]),
        "opening_boundary_contact_ratio": float(selected["boundary_contact_ratio"]),
        "rim_support_pixel_count": int(selected["rim_support_pixel_count"]),
        "evidence_score": float(selected["score"]),
        "component_count": max(0, int(count) - 1),
    }
    return {
        "eligible": True,
        "mouth_instance": mouth,
        "association": {
            "association_mode": "depth_inferred_partial_opening",
            "containment": 1.0,
            "mouth_to_ring_area_ratio": float(mouth.area_px) / float(ring_area),
            "association_score": float(selected["score"]),
        },
        "rejection_reasons": [],
        "diagnostics": diagnostics,
        "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
        "_debug": {
            "deep_candidate_mask": deep_mask,
            "opening_mask": opening_mask,
            "rim_support_mask": np.asarray(selected["rim_support_mask"], dtype=bool),
        },
    }


def _axis_candidates_constrained(
    ring_center_uv: np.ndarray,
    opening_direction_uv: np.ndarray,
    surface_depth_mm: float,
    intrinsics: Mapping[str, float],
    section: Mapping[str, Any],
) -> List[np.ndarray]:
    center_camera = np.asarray([
        (ring_center_uv[0] - float(intrinsics["cx"])) * surface_depth_mm / float(intrinsics["fx"]),
        (ring_center_uv[1] - float(intrinsics["cy"])) * surface_depth_mm / float(intrinsics["fy"]),
        surface_depth_mm,
    ], dtype=np.float64)
    view_ray = _unit(center_camera)
    image_tangent = _unit(np.asarray([
        opening_direction_uv[0] * surface_depth_mm / float(intrinsics["fx"]),
        opening_direction_uv[1] * surface_depth_mm / float(intrinsics["fy"]),
        0.0,
    ], dtype=np.float64))
    coarse = section.get("constrained_view_component_deg", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    if not isinstance(coarse, Sequence) or isinstance(coarse, (str, bytes)):
        coarse = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    candidates: List[np.ndarray] = []
    for angle_deg in coarse:
        angle = math.radians(max(0.0, min(75.0, float(angle_deg))))
        # Opening normal is constrained to have a camera-facing component.
        axis = _unit(math.cos(angle) * image_tangent + math.sin(angle) * (-view_ray))
        projected = _project_points(np.asarray([center_camera, center_camera + axis * 30.0]), intrinsics)
        if np.isfinite(projected).all() and float(np.dot(projected[1] - projected[0], opening_direction_uv)) < 0.0:
            axis = -axis
        if float(np.dot(axis, -center_camera)) <= 0.0 and float(angle_deg) > 0.0:
            axis = -axis
        candidates.append(axis)
    return candidates


def fit_partial_opening_cylinder_m383(
    ring: SegmentationInstance,
    mouth: Optional[SegmentationInstance],
    association: Optional[Mapping[str, Any]],
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
    *,
    inferred_opening: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit a projection-constrained local cylinder from segmented or depth opening evidence."""

    started = time.perf_counter()
    section = raw_config.get("m38_branch_b") or {}
    if not isinstance(section, Mapping):
        section = {}
    object_cfg = raw_config.get("object_geometry") or {}
    if not isinstance(object_cfg, Mapping):
        object_cfg = {}
    depth_cfg = raw_config.get("depth") or {}
    if not isinstance(depth_cfg, Mapping):
        depth_cfg = {}

    reasons: List[str] = []
    warnings: List[str] = []
    timing: Dict[str, float] = {}
    evidence_diag: Dict[str, Any] = {}
    source = "segmented_partial_mouth"
    if inferred_opening is not None:
        source = "depth_inferred"
        evidence_diag = dict(inferred_opening.get("diagnostics") or {})
        if not bool(inferred_opening.get("eligible", False)):
            reasons.extend(str(value) for value in inferred_opening.get("rejection_reasons") or [])
        candidate_mouth = inferred_opening.get("mouth_instance")
        if isinstance(candidate_mouth, SegmentationInstance):
            mouth = candidate_mouth
        if not association:
            association = inferred_opening.get("association") if isinstance(inferred_opening.get("association"), Mapping) else {}
    association = dict(association or {})
    if mouth is None:
        reasons.append("m383_partial_opening_evidence_unavailable")

    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    ring_center = np.asarray(ring.centroid_uv, dtype=np.float64)
    mouth_center = np.asarray(mouth.centroid_uv if mouth is not None else ring.centroid_uv, dtype=np.float64)
    opening_direction_uv = mouth_center - ring_center
    center_offset_px = float(np.linalg.norm(opening_direction_uv))
    x1, y1, x2, y2 = ring.bbox_xyxy
    bbox_diagonal = max(1.0, math.hypot(float(x2 - x1), float(y2 - y1)))
    center_offset_ratio = center_offset_px / bbox_diagonal
    minimum_offset_ratio = _safe_float(section.get("minimum_partial_opening_center_offset_ratio"), 0.10)
    if center_offset_px <= 1e-6 or center_offset_ratio < minimum_offset_ratio:
        reasons.append("m383_opening_not_off_center_enough_for_branch_b")
        opening_direction_uv = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        opening_direction_uv /= center_offset_px

    if source == "segmented_partial_mouth":
        mode = str(association.get("association_mode") or "")
        allowed = section.get("allowed_association_modes", ["strict_envelope", "bbox_fallback"])
        allowed = {str(value) for value in allowed} if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)) else {"strict_envelope", "bbox_fallback"}
        if mode not in allowed:
            reasons.append("m383_segmented_opening_association_not_supported")
        containment = _safe_float(association.get("containment"), 0.0)
        if containment < _safe_float(section.get("minimum_mouth_containment"), 0.28):
            reasons.append("m383_segmented_opening_containment_too_low")

    surface_depth = evidence_diag.get("surface_depth_mm")
    if surface_depth is None:
        surface_depth = _surface_depth_mode(
            depth_mm,
            _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 2)),
        )
    if surface_depth is None:
        reasons.append("m383_side_surface_depth_unavailable")
        surface_depth = float(np.median(depth_mm[ring.mask & (depth_mm > 0)])) if np.any(ring.mask & (depth_mm > 0)) else 1000.0

    if reasons:
        timing["evidence_gate_ms"] = (time.perf_counter() - started) * 1000.0
        timing["total_ms"] = timing["evidence_gate_ms"]
        return {
            "ring_instance_id": int(ring.instance_id),
            "mouth_instance_id": int(mouth.instance_id) if mouth is not None else None,
            "eligible": False,
            "rejection_reasons": reasons,
            "warnings": warnings,
            "association": association,
            "pose_payload": None,
            "synthetic_mouth_instance": None,
            "synthetic_ring_instance": None,
            "diagnostics": {
                "opening_source": source,
                "ring_mouth_center_offset_px": center_offset_px,
                "ring_mouth_center_offset_ratio": center_offset_ratio,
                "surface_depth_mm": float(surface_depth),
                **evidence_diag,
            },
            "timing_ms": timing,
            "_debug": dict((inferred_opening or {}).get("_debug") or {}),
        }

    prepare_started = time.perf_counter()
    assert mouth is not None
    base_mask = _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 2))
    base_mask &= ~_dilate(mouth.mask, _safe_int(section.get("mouth_exclusion_dilate_px"), 3))
    other_mask = np.zeros_like(base_mask, dtype=bool)
    for other in all_rings:
        if int(other.instance_id) != int(ring.instance_id):
            other_mask |= other.mask.astype(bool)
    base_mask &= ~_dilate(other_mask, _safe_int(section.get("neighbor_exclusion_dilate_px"), 1))
    base_mask &= (
        (depth_mm >= float(surface_depth) - _safe_float(section.get("side_depth_front_tolerance_mm"), 18.0))
        & (depth_mm <= float(surface_depth) + _safe_float(section.get("side_depth_back_tolerance_mm"), 38.0))
    )
    y_grid, x_grid = np.indices(base_mask.shape)
    inward = (
        (mouth_center[0] - x_grid.astype(np.float64)) * opening_direction_uv[0]
        + (mouth_center[1] - y_grid.astype(np.float64)) * opening_direction_uv[1]
    )
    minimum_inset = max(
        _safe_float(section.get("minimum_side_band_inset_px"), 4.0),
        bbox_diagonal * _safe_float(section.get("minimum_side_band_inset_ratio"), 0.08),
    )
    maximum_inset = max(
        minimum_inset + 4.0,
        bbox_diagonal * _safe_float(section.get("maximum_side_band_inset_ratio"), 0.60),
    )
    side_mask = base_mask & (inward >= minimum_inset) & (inward <= maximum_inset)
    depth_edges = _depth_edge_mask(
        depth_mm,
        side_mask,
        _safe_float(section.get("depth_edge_threshold_mm"), 18.0),
        _safe_int(section.get("depth_edge_dilate_px"), 1),
    )
    side_mask &= ~depth_edges
    side_mask, component_count, kept_component_count = _retain_components(
        side_mask, _safe_float(section.get("surface_component_minimum_ratio"), 0.08)
    )
    timing["side_mask_prepare_ms"] = (time.perf_counter() - prepare_started) * 1000.0

    extract_started = time.perf_counter()
    side_points, side_pixels = _deproject_mask(depth_mm, side_mask, intrinsics, minimum_depth, maximum_depth)
    normal_points, normals = _organized_normals(
        depth_mm, side_mask, intrinsics, _safe_int(section.get("normal_neighbor_step_px"), 2)
    )
    timing["side_point_and_normal_extract_ms"] = (time.perf_counter() - extract_started) * 1000.0
    if len(side_points) < _safe_int(section.get("minimum_side_points"), 80):
        reasons.append("m383_insufficient_side_surface_points")
    if len(normals) < _safe_int(section.get("minimum_normal_points"), 40):
        reasons.append("m383_insufficient_side_surface_normals")

    maximum_fit_points = max(80, _safe_int(section.get("maximum_fit_points"), 500))
    fit_points = side_points
    if len(fit_points) > maximum_fit_points:
        fit_points = fit_points[np.linspace(0, len(fit_points) - 1, maximum_fit_points).astype(np.int64)]
    maximum_normal_fit = max(40, _safe_int(section.get("maximum_normal_fit_points"), 400))
    fit_normal_points, fit_normals = normal_points, normals
    if len(fit_normals) > maximum_normal_fit:
        indexes = np.linspace(0, len(fit_normals) - 1, maximum_normal_fit).astype(np.int64)
        fit_normal_points, fit_normals = fit_normal_points[indexes], fit_normals[indexes]

    fit_started = time.perf_counter()
    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)
    inner_radius = 0.5 * _safe_float(object_cfg.get("nominal_inner_diameter_mm"), 60.0)
    axial_length = _safe_float(object_cfg.get("axial_length_mm"), 70.0)
    candidates = _axis_candidates_constrained(
        ring_center, opening_direction_uv, float(surface_depth), intrinsics, section
    )
    evaluations: List[Dict[str, Any]] = []
    if len(fit_points) >= 3:
        for axis in candidates:
            evaluations.append(_evaluate_axis(
                fit_points,
                fit_normal_points,
                fit_normals,
                axis,
                opening_direction_uv,
                intrinsics,
                outer_radius,
                _safe_int(section.get("fixed_radius_iterations"), 4),
                _safe_float(section.get("radial_inlier_threshold_mm"), 6.0),
            ))
    best = min(evaluations, key=lambda row: float(row["score"])) if evaluations else None
    timing["constrained_cylinder_fit_ms"] = (time.perf_counter() - fit_started) * 1000.0
    if best is None:
        reasons.append("m383_constrained_cylinder_fit_failed")

    synthetic_mask = np.zeros_like(ring.mask, dtype=bool)
    synthetic_outer_mask = np.zeros_like(ring.mask, dtype=bool)
    rim_support_mask = np.zeros_like(ring.mask, dtype=bool)
    opening_center = None
    far_center = None
    center_error_px = None
    synthetic_coverage = None
    observed_coverage = None
    endpoint_residual_p90 = None
    axis_view_angle_deg = None
    opening_near_margin_mm = None
    rim_point_count = 0

    anchor_started = time.perf_counter()
    if best is not None:
        axis = _unit(np.asarray(best["axis"], dtype=np.float64))
        axis_point = np.asarray(best["axis_point"], dtype=np.float64)
        projected = _project_points(np.asarray([axis_point, axis_point + axis * 30.0]), intrinsics)
        if np.isfinite(projected).all() and float(np.dot(projected[1] - projected[0], opening_direction_uv)) < 0.0:
            axis = -axis
        if float(np.dot(axis, -axis_point)) <= 0.0:
            reasons.append("m383_opening_axis_not_camera_facing")

        rim_support_mask = (
            _dilate(mouth.mask, _safe_int(section.get("rim_support_dilate_px"), 5))
            & ring.mask
            & ~mouth.mask
            & (depth_mm >= float(surface_depth) - _safe_float(section.get("side_depth_front_tolerance_mm"), 18.0))
            & (depth_mm <= float(surface_depth) + _safe_float(section.get("rim_support_depth_back_tolerance_mm"), 50.0))
        )
        rim_points, rim_pixels = _deproject_mask(depth_mm, rim_support_mask, intrinsics, minimum_depth, maximum_depth)
        rim_point_count = int(len(rim_points))
        if rim_point_count < _safe_int(section.get("minimum_rim_support_points"), 16):
            reasons.append("m383_insufficient_observed_rim_support")
        else:
            basis_u = np.asarray(best["basis_u"], dtype=np.float64)
            basis_v = np.asarray(best["basis_v"], dtype=np.float64)
            circle_center = np.asarray(best["circle_center_2d"], dtype=np.float64)
            centerline_base = basis_u * circle_center[0] + basis_v * circle_center[1]
            rim_axial = rim_points @ axis
            percentile = _safe_float(section.get("opening_rim_axial_percentile"), 90.0)
            opening_scalar = float(np.percentile(rim_axial, percentile))
            residual = np.abs(rim_axial - opening_scalar)
            threshold = _safe_float(section.get("rim_anchor_inlier_threshold_mm"), 10.0)
            inlier = residual <= threshold
            if int(np.count_nonzero(inlier)) >= _safe_int(section.get("minimum_rim_anchor_inliers"), 10):
                opening_scalar = float(np.percentile(rim_axial[inlier], percentile))
                residual = np.abs(rim_axial - opening_scalar)
                inlier = residual <= threshold
            endpoint_residual_p90 = float(np.percentile(residual[inlier], 90)) if np.any(inlier) else None
            if endpoint_residual_p90 is None or endpoint_residual_p90 > _safe_float(section.get("maximum_rim_anchor_residual_p90_mm"), 10.0):
                reasons.append("m383_observed_rim_anchor_unstable")

            opening_center = centerline_base + axis * opening_scalar
            far_center = opening_center - axis * axial_length
            opening_uv = _project_points(opening_center[None, :], intrinsics)[0]
            if not np.isfinite(opening_uv).all():
                reasons.append("m383_opening_center_not_projectable")
            else:
                center_error_px = float(np.linalg.norm(opening_uv - mouth_center))
                projected_outer = _project_points(
                    np.asarray([opening_center, opening_center + basis_u * outer_radius]), intrinsics
                )
                radius_px = float(np.linalg.norm(projected_outer[1] - projected_outer[0])) if np.isfinite(projected_outer).all() else 1.0
                maximum_error = max(
                    _safe_float(section.get("maximum_opening_center_error_px"), 20.0),
                    radius_px * _safe_float(section.get("maximum_opening_center_error_radius_ratio"), 0.80),
                )
                if center_error_px > maximum_error:
                    reasons.append("m383_opening_center_disagrees_with_observed_aperture")

            # Evaluate camera-facing visibility at the measured side-cylinder
            # centerline. The inferred rim anchor can shift the completed opening
            # center laterally enough to move an almost side-on solution across
            # 90 degrees even though the observed surface normal still faces the
            # camera. Using the measured axis point avoids that M38.2/M38.3 false
            # rejection while the aperture and rim-support gates remain active.
            view_to_camera = _unit(-axis_point)
            axis_view_angle_deg = math.degrees(
                math.acos(float(np.clip(np.dot(axis, view_to_camera), -1.0, 1.0)))
            )
            if axis_view_angle_deg < _safe_float(section.get("minimum_axis_view_angle_deg"), 35.0):
                reasons.append("m383_axis_too_frontal_for_branch_b")
            if axis_view_angle_deg > _safe_float(section.get("maximum_axis_view_angle_deg"), 89.5):
                reasons.append("m383_axis_too_side_on_for_inner_entry")
            opening_near_margin_mm = float(np.linalg.norm(far_center) - np.linalg.norm(opening_center))

            circle_u, circle_v = _basis_perpendicular(axis)
            angles = np.linspace(0.0, 2.0 * math.pi, max(64, _safe_int(section.get("synthetic_mouth_sample_count"), 96)), endpoint=False)
            circle_basis = np.cos(angles)[:, None] * circle_u[None, :] + np.sin(angles)[:, None] * circle_v[None, :]
            inner_uv = _project_points(opening_center[None, :] + inner_radius * circle_basis, intrinsics)
            outer_uv = _project_points(opening_center[None, :] + outer_radius * circle_basis, intrinsics)
            if not np.isfinite(inner_uv).all() or not np.isfinite(outer_uv).all():
                reasons.append("m383_opening_projection_failed")
            else:
                inner_u8 = np.zeros_like(ring.mask, dtype=np.uint8)
                outer_u8 = np.zeros_like(ring.mask, dtype=np.uint8)
                cv2.fillPoly(inner_u8, [np.rint(inner_uv).astype(np.int32)], 1)
                cv2.fillPoly(outer_u8, [np.rint(outer_uv).astype(np.int32)], 1)
                synthetic_mask = inner_u8.astype(bool)
                synthetic_outer_mask = outer_u8.astype(bool) | synthetic_mask
                observed_dilated = _dilate(mouth.mask, _safe_int(section.get("opening_overlap_dilate_px"), 2))
                intersection = int(np.count_nonzero(synthetic_mask & observed_dilated))
                synthetic_coverage = float(intersection) / float(max(1, np.count_nonzero(synthetic_mask)))
                observed_coverage = float(np.count_nonzero(synthetic_mask & mouth.mask)) / float(max(1, mouth.area_px))
                if synthetic_coverage < _safe_float(section.get("minimum_synthetic_opening_coverage"), 0.45):
                    reasons.append("m383_projected_opening_not_supported_by_observation")
                if int(np.count_nonzero(synthetic_mask)) < _safe_int(section.get("minimum_synthetic_mouth_area_px"), 50):
                    reasons.append("m383_projected_inner_opening_too_small")
                if int(np.count_nonzero(synthetic_outer_mask)) < _safe_int(section.get("minimum_synthetic_outer_area_px"), 100):
                    reasons.append("m383_projected_outer_opening_too_small")

        if best["radial_inlier_ratio"] < _safe_float(section.get("minimum_radial_inlier_ratio"), 0.60):
            reasons.append("m383_local_cylinder_inlier_ratio_too_low")
        if best["radial_residual_median_mm"] > _safe_float(section.get("maximum_radial_residual_median_mm"), 5.0):
            reasons.append("m383_local_cylinder_residual_median_too_high")
        if best["radial_residual_p90_mm"] > _safe_float(section.get("maximum_radial_residual_p90_mm"), 16.0):
            reasons.append("m383_local_cylinder_residual_p90_too_high")
        if best["normal_axis_median_deg"] > _safe_float(section.get("maximum_normal_axis_median_deg"), 22.0):
            reasons.append("m383_side_normals_not_perpendicular_to_axis")
        if best["normal_axis_p90_deg"] > _safe_float(section.get("maximum_normal_axis_p90_deg"), 48.0):
            reasons.append("m383_side_normal_axis_p90_too_high")
        if best["normal_radial_median_deg"] > _safe_float(section.get("maximum_normal_radial_median_deg"), 32.0):
            reasons.append("m383_side_normals_not_radial")
        if best["projected_axis_error_deg"] > _safe_float(section.get("maximum_projected_axis_error_deg"), 8.0):
            reasons.append("m383_axis_projection_disagrees_with_opening")
        if best["observed_axis_span_mm"] < _safe_float(section.get("minimum_observed_side_span_mm"), 18.0):
            reasons.append("m383_observed_side_span_too_short")

    timing["rim_anchor_and_opening_projection_ms"] = (time.perf_counter() - anchor_started) * 1000.0
    eligible = bool(
        not reasons
        and best is not None
        and opening_center is not None
        and far_center is not None
        and np.any(synthetic_mask)
        and np.any(synthetic_outer_mask)
    )
    synthetic_mouth = None
    synthetic_ring = None
    pose_payload = None
    diagnostics: Dict[str, Any] = {
        "opening_source": source,
        "surface_depth_mm": float(surface_depth),
        "ring_mouth_center_offset_px": float(center_offset_px),
        "ring_mouth_center_offset_ratio": float(center_offset_ratio),
        "side_point_count": int(len(side_points)),
        "side_normal_count": int(len(normals)),
        "side_component_count": int(component_count),
        "side_kept_component_count": int(kept_component_count),
        "candidate_axis_count": int(len(evaluations)),
        "radial_inlier_ratio": (best or {}).get("radial_inlier_ratio"),
        "radial_residual_median_mm": (best or {}).get("radial_residual_median_mm"),
        "radial_residual_p90_mm": (best or {}).get("radial_residual_p90_mm"),
        "normal_axis_median_deg": (best or {}).get("normal_axis_median_deg"),
        "normal_axis_p90_deg": (best or {}).get("normal_axis_p90_deg"),
        "normal_radial_median_deg": (best or {}).get("normal_radial_median_deg"),
        "normal_radial_p90_deg": (best or {}).get("normal_radial_p90_deg"),
        "visible_normal_span_deg": (best or {}).get("visible_normal_span_deg"),
        "projected_axis_error_deg": (best or {}).get("projected_axis_error_deg"),
        "observed_axis_span_mm": (best or {}).get("observed_axis_span_mm"),
        "rim_support_point_count": int(rim_point_count),
        "rim_anchor_residual_p90_mm": endpoint_residual_p90,
        "opening_center_error_px": center_error_px,
        "synthetic_opening_coverage": synthetic_coverage,
        "observed_opening_coverage": observed_coverage,
        "opening_near_margin_mm": opening_near_margin_mm,
        "axis_view_angle_deg": axis_view_angle_deg,
        "synthetic_mouth_area_px": int(np.count_nonzero(synthetic_mask)),
        "synthetic_outer_area_px": int(np.count_nonzero(synthetic_outer_mask)),
        **evidence_diag,
    }

    if eligible and best is not None and opening_center is not None and far_center is not None:
        synthetic_mouth = SegmentationInstance(
            instance_id=int(mouth.instance_id), class_id=int(mouth.class_id), class_name="ring_mouth",
            confidence=float(mouth.confidence), mask=synthetic_mask, bbox_xyxy=_bbox_from_mask(synthetic_mask)
        )
        # Preserve the measured full foam-ring instance and only add the nominal
        # projected end outline. Replacing the entire ring with the small end
        # disc (M38.2 behavior) made valid side targets fail ``ring_area_too_small``
        # before the rim-pinch/collision evaluator could inspect them.
        completed_ring_mask = ring.mask.astype(bool) | synthetic_outer_mask
        synthetic_ring = SegmentationInstance(
            instance_id=int(ring.instance_id), class_id=int(ring.class_id), class_name="foam_ring",
            confidence=float(ring.confidence), mask=completed_ring_mask, bbox_xyxy=_bbox_from_mask(completed_ring_mask)
        )
        axis = _unit(np.asarray(best["axis"], dtype=np.float64))
        projected = _project_points(np.asarray([opening_center, opening_center + axis * 30.0]), intrinsics)
        if np.isfinite(projected).all() and float(np.dot(projected[1] - projected[0], opening_direction_uv)) < 0.0:
            axis = -axis
        pose_payload = {
            "ring_instance_id": int(ring.instance_id),
            "mouth_instance_id": int(mouth.instance_id),
            "normal_toward_camera": axis.tolist(),
            "opening_center_camera_mm": np.asarray(opening_center, dtype=np.float64).tolist(),
            "far_opening_center_camera_mm": np.asarray(far_center, dtype=np.float64).tolist(),
            "plane_offset": float(-np.dot(axis, opening_center)),
            "side_point_count": int(len(side_points)),
            "side_plane_inlier_ratio": float(best["radial_inlier_ratio"]),
            "side_residual_median_mm": float(best["radial_residual_median_mm"]),
            "side_residual_p95_mm": float(best["radial_residual_p90_mm"]),
            "diagnostics": {
                **diagnostics,
                "opening_partial": True,
                "pose_source": "depth_or_segmented_partial_opening_constrained_cylinder",
                "model_completed_opening": True,
            },
        }

    timing["total_ms"] = (time.perf_counter() - started) * 1000.0
    debug = {
        "side_surface_mask": side_mask,
        "depth_edge_mask": depth_edges,
        "endpoint_support_mask": rim_support_mask,
        "synthetic_mouth_mask": synthetic_mask,
        "synthetic_outer_mask": synthetic_outer_mask,
        "side_points_camera_mm": side_points,
        "side_pixels": side_pixels,
    }
    debug.update(dict((inferred_opening or {}).get("_debug") or {}))
    return {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "eligible": bool(eligible),
        "rejection_reasons": reasons,
        "warnings": warnings,
        "association": association,
        "pose_payload": pose_payload,
        "synthetic_mouth_instance": synthetic_mouth,
        "synthetic_ring_instance": synthetic_ring,
        "diagnostics": diagnostics,
        "timing_ms": timing,
        "_debug": debug,
    }
