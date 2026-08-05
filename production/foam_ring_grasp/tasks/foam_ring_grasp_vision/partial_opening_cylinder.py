"""M38.2 branch-B partial-opening plus local outer-cylinder pose recovery.

The branch is intentionally narrower than the retained M37.6 global hollow-
cylinder search.  It requires a segmented, but incomplete, ``ring_mouth`` near
one axial end.  Only the observed outer side patch behind that mouth is fitted
to the known outer radius.  The partial end support then anchors the opening
plane.  Nominal inner and outer opening projections are synthesized only
after all geometric gates pass.  They let the existing rim-pinch and collision
pipeline recover the measured wall section without pretending that an
entirely hidden opening was observed.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .segmentation import SegmentationInstance
from .side_ring_template import (
    _basis_perpendicular,
    _fit_circle_center_fixed_radius,
    _local_axis_candidates,
    _unit,
)

_EPS = 1e-9


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _kernel(radius: int) -> np.ndarray:
    size = max(1, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return cv2.erode(mask.astype(np.uint8), _kernel(radius), iterations=1).astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius), iterations=1).astype(bool)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _project_points(points: np.ndarray, intrinsics: Mapping[str, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    z = points[:, 2]
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    if np.any(valid):
        result[valid, 0] = float(intrinsics["fx"]) * points[valid, 0] / z[valid] + float(intrinsics["cx"])
        result[valid, 1] = float(intrinsics["fy"]) * points[valid, 1] / z[valid] + float(intrinsics["cy"])
    return result


def _deproject_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: Mapping[str, float],
    minimum_mm: float,
    maximum_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & (depth_mm >= minimum_mm) & (depth_mm <= maximum_mm)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    z = depth_mm[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (ys.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.column_stack((x, y, z)), np.column_stack((xs, ys)).astype(np.int32)


def _depth_edge_mask(
    depth_mm: np.ndarray,
    candidate_mask: np.ndarray,
    threshold_mm: float,
    dilate_px: int,
) -> np.ndarray:
    depth = depth_mm.astype(np.float64, copy=False)
    edge = np.zeros(candidate_mask.shape, dtype=bool)
    horizontal = (
        candidate_mask[:, :-1]
        & candidate_mask[:, 1:]
        & (np.abs(depth[:, :-1] - depth[:, 1:]) > float(threshold_mm))
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    vertical = (
        candidate_mask[:-1, :]
        & candidate_mask[1:, :]
        & (np.abs(depth[:-1, :] - depth[1:, :]) > float(threshold_mm))
    )
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    if dilate_px > 0 and np.any(edge):
        edge = _dilate(edge, dilate_px)
    return edge


def _retain_components(mask: np.ndarray, minimum_ratio: float) -> Tuple[np.ndarray, int, int]:
    if not np.any(mask):
        return mask.astype(bool), 0, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component_count = max(0, int(count) - 1)
    if component_count <= 0:
        return mask.astype(bool), component_count, 0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(np.max(areas))
    minimum = max(8, int(round(largest * float(minimum_ratio))))
    keep = [index + 1 for index, area in enumerate(areas) if int(area) >= minimum]
    return np.isin(labels, keep), component_count, len(keep)


def _organized_normals(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: Mapping[str, float],
    step_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    step = max(1, int(step_px))
    height, width = depth_mm.shape[:2]
    if height <= 2 * step or width <= 2 * step:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    grid_y, grid_x = np.indices((height, width))
    z = depth_mm.astype(np.float64, copy=False)
    point_map = np.empty((height, width, 3), dtype=np.float64)
    point_map[..., 2] = z
    point_map[..., 0] = (grid_x - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    point_map[..., 1] = (grid_y - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    center_mask = mask[step:-step, step:-step]
    valid = (
        center_mask
        & mask[step:-step, :-2 * step]
        & mask[step:-step, 2 * step:]
        & mask[:-2 * step, step:-step]
        & mask[2 * step:, step:-step]
    )
    dx = point_map[step:-step, 2 * step:] - point_map[step:-step, :-2 * step]
    dy = point_map[2 * step:, step:-step] - point_map[:-2 * step, step:-step]
    normal_map = np.cross(dx, dy)
    norm = np.linalg.norm(normal_map, axis=2)
    valid &= np.isfinite(norm) & (norm > 1e-6)
    points = point_map[step:-step, step:-step][valid]
    normals = normal_map[valid] / norm[valid, None]
    if len(normals):
        flip = np.sum(normals * (-points), axis=1) < 0.0
        normals[flip] *= -1.0
    return points.astype(np.float64), normals.astype(np.float64)


def _normal_covariance_axis(normals: np.ndarray) -> Optional[np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(normals) < 3:
        return None
    covariance = normals.T @ normals / float(len(normals))
    try:
        values, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None
    axis = vectors[:, int(np.argmin(values))]
    try:
        return _unit(axis)
    except ValueError:
        return None


def _axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(float(np.dot(_unit(first), _unit(second)))), 0.0, 1.0)))
    )


def _projected_axis_angle_deg(
    axis: np.ndarray,
    center: np.ndarray,
    opening_direction_uv: np.ndarray,
    intrinsics: Mapping[str, float],
) -> float:
    samples = _project_points(
        np.asarray([center - axis * 25.0, center + axis * 25.0], dtype=np.float64),
        intrinsics,
    )
    if not np.isfinite(samples).all():
        return 90.0
    vector = samples[1] - samples[0]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return 90.0
    vector /= norm
    return math.degrees(
        math.acos(float(np.clip(abs(float(np.dot(vector, opening_direction_uv))), 0.0, 1.0)))
    )


def _occupied_span_deg(angles: np.ndarray) -> float:
    values = np.mod(np.asarray(angles, dtype=np.float64), 2.0 * math.pi)
    if len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * math.pi)))
    return math.degrees(max(0.0, 2.0 * math.pi - float(np.max(gaps))))


def _evaluate_axis(
    points: np.ndarray,
    normal_points: np.ndarray,
    normals: np.ndarray,
    axis: np.ndarray,
    opening_direction_uv: np.ndarray,
    intrinsics: Mapping[str, float],
    outer_radius_mm: float,
    fixed_radius_iterations: int,
    radial_inlier_threshold_mm: float,
) -> Dict[str, Any]:
    axis = _unit(axis)
    basis_u, basis_v = _basis_perpendicular(axis)
    projected = np.column_stack((points @ basis_u, points @ basis_v))
    circle_center = _fit_circle_center_fixed_radius(
        projected, float(outer_radius_mm), int(fixed_radius_iterations)
    )
    radial_distance = np.linalg.norm(projected - circle_center[None, :], axis=1)
    residual = np.abs(radial_distance - float(outer_radius_mm))
    inlier = residual <= float(radial_inlier_threshold_mm)
    axial = points @ axis
    span_values = axial[inlier] if int(np.count_nonzero(inlier)) >= 20 else axial
    observed_span = (
        float(np.percentile(span_values, 95) - np.percentile(span_values, 5))
        if len(span_values)
        else 0.0
    )
    median_axial = float(np.median(span_values)) if len(span_values) else float(np.median(axial))
    axis_point = basis_u * float(circle_center[0]) + basis_v * float(circle_center[1]) + axis * median_axial

    normal_axis_error = np.empty((0,), dtype=np.float64)
    normal_radial_error = np.empty((0,), dtype=np.float64)
    visible_span = 0.0
    if len(normals):
        normal_axis_error = np.degrees(np.arcsin(np.clip(np.abs(normals @ axis), 0.0, 1.0)))
        normal_projected = np.column_stack((normal_points @ basis_u, normal_points @ basis_v))
        radial_2d = normal_projected - circle_center[None, :]
        radial_norm = np.linalg.norm(radial_2d, axis=1)
        valid = radial_norm > 1e-6
        radial_unit = np.zeros_like(radial_2d)
        radial_unit[valid] = radial_2d[valid] / radial_norm[valid, None]
        normals_2d = np.column_stack((normals @ basis_u, normals @ basis_v))
        normals_2d_norm = np.linalg.norm(normals_2d, axis=1)
        valid &= normals_2d_norm > 1e-6
        normalized_normals = np.zeros_like(normals_2d)
        normalized_normals[valid] = normals_2d[valid] / normals_2d_norm[valid, None]
        alignment = np.abs(np.sum(normalized_normals * radial_unit, axis=1))
        normal_radial_error = np.degrees(np.arccos(np.clip(alignment[valid], 0.0, 1.0)))
        angles = np.arctan2(radial_2d[valid, 1], radial_2d[valid, 0])
        visible_span = _occupied_span_deg(angles)

    radial_ratio = float(np.mean(inlier)) if len(inlier) else 0.0
    radial_median = float(np.median(residual)) if len(residual) else float("inf")
    radial_p90 = float(np.percentile(residual, 90)) if len(residual) else float("inf")
    axis_normal_median = float(np.median(normal_axis_error)) if len(normal_axis_error) else 90.0
    axis_normal_p90 = float(np.percentile(normal_axis_error, 90)) if len(normal_axis_error) else 90.0
    radial_normal_median = float(np.median(normal_radial_error)) if len(normal_radial_error) else 90.0
    radial_normal_p90 = float(np.percentile(normal_radial_error, 90)) if len(normal_radial_error) else 90.0
    image_error = _projected_axis_angle_deg(axis, axis_point, opening_direction_uv, intrinsics)
    score = (
        radial_median
        + 0.22 * radial_p90
        + 7.0 * (1.0 - radial_ratio)
        + 0.045 * axis_normal_median
        + 0.025 * radial_normal_median
        + 0.025 * image_error
    )
    return {
        "score": float(score),
        "axis": axis,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "circle_center_2d": circle_center,
        "axis_point": axis_point,
        "radial_inlier_mask": inlier,
        "radial_inlier_ratio": radial_ratio,
        "radial_residual_median_mm": radial_median,
        "radial_residual_p90_mm": radial_p90,
        "normal_axis_median_deg": axis_normal_median,
        "normal_axis_p90_deg": axis_normal_p90,
        "normal_radial_median_deg": radial_normal_median,
        "normal_radial_p90_deg": radial_normal_p90,
        "visible_normal_span_deg": float(visible_span),
        "projected_axis_error_deg": float(image_error),
        "observed_axis_span_mm": float(observed_span),
        "axial_coordinate_mm": axial,
    }


def _deduplicate_axes(axes: Sequence[np.ndarray], minimum_angle_deg: float = 3.0) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for raw in axes:
        try:
            axis = _unit(raw)
        except ValueError:
            continue
        if any(_axis_angle_deg(axis, existing) < float(minimum_angle_deg) for existing in result):
            continue
        result.append(axis)
    return result


def fit_partial_opening_cylinder(
    ring: SegmentationInstance,
    mouth: SegmentationInstance,
    association: Mapping[str, Any],
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recover one near opening from partial mouth and local side evidence."""

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
    association_mode = str(association.get("association_mode") or "")
    allowed_modes = section.get("allowed_association_modes", ["strict_envelope", "bbox_fallback"])
    allowed_modes = {str(value) for value in allowed_modes} if isinstance(allowed_modes, Sequence) and not isinstance(allowed_modes, (str, bytes)) else {"strict_envelope", "bbox_fallback"}
    if association_mode not in allowed_modes:
        reasons.append("m38b_opening_association_not_supported")
    containment = _safe_float(association.get("containment"), 0.0)
    area_ratio = _safe_float(association.get("mouth_to_ring_area_ratio"), 0.0)
    if containment < _safe_float(section.get("minimum_mouth_containment"), 0.28):
        reasons.append("m38b_mouth_containment_too_low")
    if not (
        _safe_float(section.get("minimum_mouth_to_ring_area_ratio"), 0.012)
        <= area_ratio
        <= _safe_float(section.get("maximum_mouth_to_ring_area_ratio"), 0.50)
    ):
        reasons.append("m38b_mouth_area_ratio_out_of_range")

    ring_center = np.asarray(ring.centroid_uv, dtype=np.float64)
    mouth_center = np.asarray(mouth.centroid_uv, dtype=np.float64)
    opening_direction_uv = mouth_center - ring_center
    center_offset_px = float(np.linalg.norm(opening_direction_uv))
    if center_offset_px <= _safe_float(section.get("minimum_ring_mouth_center_offset_px"), 3.0):
        reasons.append("m38b_partial_opening_endpoint_direction_unavailable")
        opening_direction_uv = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        opening_direction_uv /= center_offset_px

    # These are evidence-availability gates, not fit-quality gates.  When the
    # segmented mouth is not plausibly attached to one visible axial end there
    # is no reason to spend time on local cylinder optimization.
    if reasons:
        timing["evidence_gate_ms"] = (time.perf_counter() - started) * 1000.0
        timing["total_ms"] = timing["evidence_gate_ms"]
        return {
            "ring_instance_id": int(ring.instance_id),
            "mouth_instance_id": int(mouth.instance_id),
            "eligible": False,
            "rejection_reasons": reasons,
            "warnings": warnings,
            "association": dict(association),
            "pose_payload": None,
            "synthetic_mouth_instance": None,
            "synthetic_ring_instance": None,
            "diagnostics": {
                "opening_partial": True,
                "association_mode": association_mode,
                "mouth_containment": float(containment),
                "mouth_to_ring_area_ratio": float(area_ratio),
                "ring_mouth_center_offset_px": float(center_offset_px),
                "side_point_count": 0,
                "side_normal_count": 0,
                "endpoint_support_point_count": 0,
                "candidate_axis_count": 0,
            },
            "timing_ms": timing,
            "_debug": {},
        }

    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    prepare_started = time.perf_counter()
    base_mask = _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 2))
    base_mask &= ~_dilate(mouth.mask, _safe_int(section.get("mouth_exclusion_dilate_px"), 3))
    other_mask = np.zeros_like(base_mask, dtype=bool)
    for other in all_rings:
        if int(other.instance_id) != int(ring.instance_id):
            other_mask |= other.mask.astype(bool)
    other_mask = _dilate(other_mask, _safe_int(section.get("neighbor_exclusion_dilate_px"), 2))
    base_mask &= ~other_mask
    base_mask &= (depth_mm >= minimum_depth) & (depth_mm <= maximum_depth)

    y_grid, x_grid = np.indices(base_mask.shape)
    inward = (
        (mouth_center[0] - x_grid.astype(np.float64)) * opening_direction_uv[0]
        + (mouth_center[1] - y_grid.astype(np.float64)) * opening_direction_uv[1]
    )
    x1, y1, x2, y2 = ring.bbox_xyxy
    bbox_diagonal = max(1.0, math.hypot(float(x2 - x1), float(y2 - y1)))
    minimum_inset = max(
        _safe_float(section.get("minimum_side_band_inset_px"), 4.0),
        bbox_diagonal * _safe_float(section.get("minimum_side_band_inset_ratio"), 0.05),
    )
    maximum_inset = max(
        minimum_inset + 4.0,
        bbox_diagonal * _safe_float(section.get("maximum_side_band_inset_ratio"), 0.72),
    )
    side_mask = base_mask & (inward >= minimum_inset) & (inward <= maximum_inset)
    depth_edges = _depth_edge_mask(
        depth_mm,
        side_mask,
        _safe_float(section.get("depth_edge_threshold_mm"), 14.0),
        _safe_int(section.get("depth_edge_dilate_px"), 1),
    )
    side_mask &= ~depth_edges
    side_mask, component_count, kept_component_count = _retain_components(
        side_mask, _safe_float(section.get("surface_component_minimum_ratio"), 0.08)
    )
    timing["side_mask_prepare_ms"] = (time.perf_counter() - prepare_started) * 1000.0

    extract_started = time.perf_counter()
    side_points, side_pixels = _deproject_mask(
        depth_mm, side_mask, intrinsics, minimum_depth, maximum_depth
    )
    normal_points, normals = _organized_normals(
        depth_mm,
        side_mask,
        intrinsics,
        _safe_int(section.get("normal_neighbor_step_px"), 2),
    )
    timing["side_point_and_normal_extract_ms"] = (time.perf_counter() - extract_started) * 1000.0
    minimum_side_points = _safe_int(section.get("minimum_side_points"), 80)
    minimum_normal_points = _safe_int(section.get("minimum_normal_points"), 40)
    if len(side_points) < minimum_side_points:
        reasons.append("m38b_insufficient_side_surface_points")
    if len(normals) < minimum_normal_points:
        reasons.append("m38b_insufficient_side_surface_normals")

    maximum_fit_points = max(80, _safe_int(section.get("maximum_fit_points"), 600))
    if len(side_points) > maximum_fit_points:
        indexes = np.linspace(0, len(side_points) - 1, maximum_fit_points).astype(np.int64)
        fit_points = side_points[indexes]
    else:
        fit_points = side_points
    maximum_normal_fit = max(40, _safe_int(section.get("maximum_normal_fit_points"), 500))
    if len(normals) > maximum_normal_fit:
        indexes = np.linspace(0, len(normals) - 1, maximum_normal_fit).astype(np.int64)
        fit_normal_points = normal_points[indexes]
        fit_normals = normals[indexes]
    else:
        fit_normal_points = normal_points
        fit_normals = normals

    fit_started = time.perf_counter()
    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)
    inner_radius = 0.5 * _safe_float(object_cfg.get("nominal_inner_diameter_mm"), 60.0)
    axial_length = _safe_float(object_cfg.get("axial_length_mm"), 70.0)
    seeds: List[np.ndarray] = []
    covariance_axis = _normal_covariance_axis(fit_normals)
    if covariance_axis is not None:
        seeds.append(covariance_axis)
    if len(fit_points) >= 3:
        centered = fit_points - np.mean(fit_points, axis=0)
        try:
            _values, vectors = np.linalg.eigh(centered.T @ centered)
            seeds.append(vectors[:, int(np.argmax(_values))])
        except np.linalg.LinAlgError:
            pass
    median_depth = float(np.median(fit_points[:, 2])) if len(fit_points) else 1000.0
    image_axis_seed = np.asarray(
        [
            opening_direction_uv[0] * median_depth / float(intrinsics["fx"]),
            opening_direction_uv[1] * median_depth / float(intrinsics["fy"]),
            0.0,
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(image_axis_seed)) > 1e-6:
        seeds.append(image_axis_seed)
    seeds = _deduplicate_axes(seeds)

    refine_angles = section.get("local_refine_angles_deg", [7.0, 2.5])
    if not isinstance(refine_angles, Sequence) or isinstance(refine_angles, (str, bytes)):
        refine_angles = [7.0, 2.5]
    refine_angles = [float(value) for value in refine_angles]
    radial_steps = max(1, _safe_int(section.get("local_refine_radial_steps"), 1))
    azimuth_steps = max(4, _safe_int(section.get("local_refine_azimuth_steps"), 8))
    refine_top_k = max(1, _safe_int(section.get("local_refine_top_k"), 3))
    fixed_radius_iterations = _safe_int(section.get("fixed_radius_iterations"), 6)
    radial_inlier_threshold = _safe_float(
        section.get("radial_inlier_threshold_mm"), 6.0
    )

    candidate_axes: List[np.ndarray] = []
    evaluations: List[Dict[str, Any]] = []

    def evaluate_new_axes(raw_axes: Sequence[np.ndarray]) -> None:
        nonlocal candidate_axes, evaluations
        unique = _deduplicate_axes(
            [*candidate_axes, *raw_axes], minimum_angle_deg=0.35
        )
        new_axes = unique[len(candidate_axes) :]
        candidate_axes = unique
        if len(fit_points) < 3:
            return
        for candidate_axis in new_axes:
            evaluations.append(
                _evaluate_axis(
                    fit_points,
                    fit_normal_points,
                    fit_normals,
                    candidate_axis,
                    opening_direction_uv,
                    intrinsics,
                    outer_radius,
                    fixed_radius_iterations,
                    radial_inlier_threshold,
                )
            )

    evaluate_new_axes(seeds)
    active_axes = list(seeds)
    for angle in refine_angles:
        level_axes: List[np.ndarray] = []
        for current in active_axes:
            level_axes.extend(
                _local_axis_candidates(
                    current, float(angle), radial_steps, azimuth_steps
                )
            )
        evaluate_new_axes(level_axes)
        if evaluations:
            active_axes = [
                np.asarray(row["axis"], dtype=np.float64)
                for row in sorted(
                    evaluations, key=lambda row: float(row["score"])
                )[:refine_top_k]
            ]
        else:
            active_axes = []
    best = min(evaluations, key=lambda row: float(row["score"])) if evaluations else None
    timing["local_cylinder_fit_ms"] = (time.perf_counter() - fit_started) * 1000.0
    if best is None:
        reasons.append("m38b_local_cylinder_fit_failed")

    opening_center = None
    far_center = None
    endpoint_support_mask = np.zeros_like(side_mask, dtype=bool)
    endpoint_points = np.empty((0, 3), dtype=np.float64)
    endpoint_pixels = np.empty((0, 2), dtype=np.int32)
    synthetic_mask = np.zeros_like(side_mask, dtype=bool)
    synthetic_outer_mask = np.zeros_like(side_mask, dtype=bool)
    endpoint_axial_p90 = None
    endpoint_inlier_count = 0
    endpoint_inlier_ratio = 0.0
    mouth_center_error_px = None
    mouth_overlap_ratio = None
    opening_near_margin_mm = None
    axis_view_angle_deg = None
    if best is not None:
        axis = np.asarray(best["axis"], dtype=np.float64)
        axis_point = np.asarray(best["axis_point"], dtype=np.float64)
        endpoint_candidates = np.asarray(
            [axis_point - axis * (0.5 * axial_length), axis_point + axis * (0.5 * axial_length)],
            dtype=np.float64,
        )
        endpoint_uv = _project_points(endpoint_candidates, intrinsics)
        if np.isfinite(endpoint_uv).all():
            distances = np.linalg.norm(endpoint_uv - mouth_center[None, :], axis=1)
            if int(np.argmin(distances)) == 0:
                axis = -axis
        # axis now points from the cylinder body toward the observed opening.
        if float(np.dot(axis, -axis_point)) < 0.0:
            # Keep the mouth end, rather than camera direction, authoritative;
            # this diagnostic is checked again after the endpoint is anchored.
            warnings.append("m38b_axis_toward_opening_not_initially_camera_facing")

        mouth_equivalent_radius = math.sqrt(max(1.0, float(mouth.area_px)) / math.pi)
        endpoint_expand = max(
            _safe_int(section.get("minimum_endpoint_support_expand_px"), 6),
            int(round(mouth_equivalent_radius * _safe_float(section.get("endpoint_support_expand_ratio"), 0.70))),
        )
        endpoint_expand = min(
            _safe_int(section.get("maximum_endpoint_support_expand_px"), 28),
            endpoint_expand,
        )
        endpoint_support_mask = (
            _erode(ring.mask, _safe_int(section.get("endpoint_ring_erode_px"), 1))
            & _dilate(mouth.mask, endpoint_expand)
            & ~_dilate(mouth.mask, _safe_int(section.get("endpoint_mouth_inner_exclusion_px"), 1))
            & ~other_mask
            & ~depth_edges
        )
        endpoint_points, endpoint_pixels = _deproject_mask(
            depth_mm, endpoint_support_mask, intrinsics, minimum_depth, maximum_depth
        )
        if len(endpoint_points) < _safe_int(section.get("minimum_endpoint_support_points"), 20):
            reasons.append("m38b_insufficient_partial_end_support")
        else:
            basis_u = np.asarray(best["basis_u"], dtype=np.float64)
            basis_v = np.asarray(best["basis_v"], dtype=np.float64)
            circle_center = np.asarray(best["circle_center_2d"], dtype=np.float64)
            centerline_base = basis_u * circle_center[0] + basis_v * circle_center[1]
            endpoint_axial = endpoint_points @ axis
            # The expanded mouth neighbourhood intentionally contains some
            # curved side-wall points.  Fit only the axial extreme toward the
            # observed mouth instead of requiring the entire neighbourhood to
            # be planar.  This is a one-dimensional robust end-plane fit, not a
            # hidden full-end reconstruction.
            opening_seed = float(
                np.percentile(
                    endpoint_axial,
                    _safe_float(section.get("opening_plane_axial_percentile"), 78.0),
                )
            )
            threshold = _safe_float(
                section.get("endpoint_plane_inlier_threshold_mm"), 8.0
            )
            extreme = endpoint_axial >= opening_seed
            extreme_values = endpoint_axial[extreme]
            opening_scalar = (
                float(np.median(extreme_values))
                if len(extreme_values)
                else opening_seed
            )
            residual = np.abs(endpoint_axial - opening_scalar)
            inlier = residual <= threshold
            if int(np.count_nonzero(inlier)) >= max(
                10, _safe_int(section.get("minimum_endpoint_support_points"), 20)
            ):
                opening_scalar = float(np.median(endpoint_axial[inlier]))
                residual = np.abs(endpoint_axial - opening_scalar)
                inlier = residual <= threshold
            endpoint_inlier_count = int(np.count_nonzero(inlier))
            endpoint_inlier_ratio = float(endpoint_inlier_count) / float(
                max(1, len(endpoint_points))
            )
            endpoint_axial_p90 = (
                float(np.percentile(residual[inlier], 90))
                if endpoint_inlier_count
                else None
            )
            if endpoint_inlier_count < _safe_int(
                section.get("minimum_endpoint_plane_inlier_points"), 20
            ):
                reasons.append("m38b_partial_end_plane_inlier_count_too_low")
            if endpoint_inlier_ratio < _safe_float(
                section.get("minimum_endpoint_plane_inlier_ratio"), 0.08
            ):
                reasons.append("m38b_partial_end_plane_inlier_ratio_too_low")
            opening_center = centerline_base + axis * opening_scalar
            far_center = opening_center - axis * axial_length
            opening_uv = _project_points(opening_center[None, :], intrinsics)[0]
            if not np.isfinite(opening_uv).all():
                reasons.append("m38b_opening_center_not_projectable")
            else:
                mouth_center_error_px = float(np.linalg.norm(opening_uv - mouth_center))
                projected_outer = _project_points(
                    np.asarray([opening_center, opening_center + basis_u * outer_radius]), intrinsics
                )
                radius_px = float(np.linalg.norm(projected_outer[1] - projected_outer[0])) if np.isfinite(projected_outer).all() else 1.0
                maximum_error = max(
                    _safe_float(section.get("maximum_opening_center_error_px"), 18.0),
                    radius_px * _safe_float(section.get("maximum_opening_center_error_radius_ratio"), 0.55),
                )
                if mouth_center_error_px > maximum_error:
                    reasons.append("m38b_partial_mouth_disagrees_with_cylinder_endpoint")

            if (
                endpoint_axial_p90 is None
                or endpoint_axial_p90
                > _safe_float(
                    section.get("maximum_endpoint_axial_residual_p90_mm"), 8.0
                )
            ):
                reasons.append("m38b_partial_end_support_not_planar_along_axis")

            camera_open = float(np.linalg.norm(opening_center))
            camera_far = float(np.linalg.norm(far_center))
            opening_near_margin_mm = camera_far - camera_open
            if opening_near_margin_mm < _safe_float(section.get("minimum_opening_near_margin_mm"), -3.0):
                reasons.append("m38b_partial_opening_is_not_camera_near_endpoint")
            if float(np.dot(axis, -opening_center)) <= 0.0:
                reasons.append("m38b_opening_axis_not_toward_camera")
            view_toward_camera = _unit(-opening_center)
            axis_view_angle_deg = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(float(np.dot(_unit(axis), view_toward_camera))),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            if axis_view_angle_deg < _safe_float(
                section.get("minimum_axis_view_angle_deg"), 24.0
            ):
                reasons.append("m38b_axis_too_frontal_for_partial_opening_branch")
            if axis_view_angle_deg > _safe_float(
                section.get("maximum_axis_view_angle_deg"), 88.0
            ):
                reasons.append("m38b_axis_too_side_on_for_reliable_inner_opening_entry")

            circle_u, circle_v = _basis_perpendicular(axis)
            circle_angles = np.linspace(
                0.0,
                2.0 * math.pi,
                max(48, _safe_int(section.get("synthetic_mouth_sample_count"), 96)),
                endpoint=False,
            )
            circle_basis = (
                np.cos(circle_angles)[:, None] * circle_u[None, :]
                + np.sin(circle_angles)[:, None] * circle_v[None, :]
            )
            inner_circle_uv = _project_points(
                opening_center[None, :] + inner_radius * circle_basis,
                intrinsics,
            )
            outer_circle_uv = _project_points(
                opening_center[None, :] + outer_radius * circle_basis,
                intrinsics,
            )
            inner_finite = np.isfinite(inner_circle_uv).all(axis=1)
            outer_finite = np.isfinite(outer_circle_uv).all(axis=1)
            if (
                int(np.count_nonzero(inner_finite)) < 24
                or int(np.count_nonzero(outer_finite)) < 24
            ):
                reasons.append("m38b_synthetic_opening_projection_failed")
            else:
                inner_polygon = np.rint(inner_circle_uv[inner_finite]).astype(np.int32)
                outer_polygon = np.rint(outer_circle_uv[outer_finite]).astype(np.int32)
                synthetic_u8 = np.zeros_like(synthetic_mask, dtype=np.uint8)
                synthetic_outer_u8 = np.zeros_like(synthetic_outer_mask, dtype=np.uint8)
                cv2.fillPoly(synthetic_u8, [inner_polygon], 1)
                cv2.fillPoly(synthetic_outer_u8, [outer_polygon], 1)
                synthetic_mask = synthetic_u8.astype(bool)
                synthetic_outer_mask = synthetic_outer_u8.astype(bool)
                # Numerical projection can occasionally leave one or two inner
                # pixels outside the rasterized outer disk.  The outer envelope
                # must contain the complete nominal opening for deterministic
                # ray-based wall-thickness evaluation.
                synthetic_outer_mask |= synthetic_mask
                original_area = max(1, int(mouth.area_px))
                mouth_overlap_ratio = float(
                    np.count_nonzero(synthetic_mask & mouth.mask)
                ) / float(original_area)
                if mouth_overlap_ratio < _safe_float(
                    section.get("minimum_partial_mouth_overlap_ratio"), 0.45
                ):
                    reasons.append(
                        "m38b_partial_mouth_overlap_with_fitted_opening_too_low"
                    )
                if int(np.count_nonzero(synthetic_mask)) < _safe_int(
                    section.get("minimum_synthetic_mouth_area_px"), 80
                ):
                    reasons.append("m38b_synthetic_opening_too_small")
                if int(np.count_nonzero(synthetic_outer_mask)) < _safe_int(
                    section.get("minimum_synthetic_outer_area_px"), 180
                ):
                    reasons.append("m38b_synthetic_outer_opening_too_small")

        if best["radial_inlier_ratio"] < _safe_float(section.get("minimum_radial_inlier_ratio"), 0.55):
            reasons.append("m38b_local_cylinder_inlier_ratio_too_low")
        if best["radial_residual_median_mm"] > _safe_float(section.get("maximum_radial_residual_median_mm"), 4.5):
            reasons.append("m38b_local_cylinder_residual_median_too_high")
        if best["radial_residual_p90_mm"] > _safe_float(section.get("maximum_radial_residual_p90_mm"), 13.0):
            reasons.append("m38b_local_cylinder_residual_p90_too_high")
        if best["normal_axis_median_deg"] > _safe_float(section.get("maximum_normal_axis_median_deg"), 16.0):
            reasons.append("m38b_side_normals_not_perpendicular_to_axis")
        if best["normal_axis_p90_deg"] > _safe_float(section.get("maximum_normal_axis_p90_deg"), 38.0):
            reasons.append("m38b_side_normal_axis_p90_too_high")
        if best["normal_radial_median_deg"] > _safe_float(section.get("maximum_normal_radial_median_deg"), 28.0):
            reasons.append("m38b_side_normals_not_radial")
        if best["visible_normal_span_deg"] < _safe_float(section.get("minimum_visible_normal_span_deg"), 20.0):
            reasons.append("m38b_visible_cylinder_arc_too_small")
        if best["projected_axis_error_deg"] > _safe_float(section.get("maximum_projected_axis_error_deg"), 28.0):
            reasons.append("m38b_axis_projection_disagrees_with_partial_opening")
        if best["observed_axis_span_mm"] < _safe_float(section.get("minimum_observed_side_span_mm"), 18.0):
            reasons.append("m38b_observed_side_span_too_short")

    timing["endpoint_anchor_and_synthetic_mouth_ms"] = max(
        0.0,
        (time.perf_counter() - started) * 1000.0
        - sum(float(value) for value in timing.values()),
    )
    eligible = bool(
        not reasons
        and best is not None
        and opening_center is not None
        and far_center is not None
        and np.any(synthetic_mask)
        and np.any(synthetic_outer_mask)
    )
    synthetic_instance: Optional[SegmentationInstance] = None
    synthetic_ring_instance: Optional[SegmentationInstance] = None
    pose_payload: Optional[Dict[str, Any]] = None
    if eligible and best is not None and opening_center is not None and far_center is not None:
        synthetic_instance = SegmentationInstance(
            instance_id=int(mouth.instance_id),
            class_id=int(mouth.class_id),
            class_name=str(mouth.class_name),
            confidence=float(mouth.confidence),
            mask=synthetic_mask.astype(bool),
            bbox_xyxy=_bbox_from_mask(synthetic_mask),
        )
        synthetic_ring_instance = SegmentationInstance(
            instance_id=int(ring.instance_id),
            class_id=int(ring.class_id),
            class_name=str(ring.class_name),
            confidence=float(ring.confidence),
            mask=synthetic_outer_mask.astype(bool),
            bbox_xyxy=_bbox_from_mask(synthetic_outer_mask),
        )
        axis = _unit(np.asarray(best["axis"], dtype=np.float64))
        # Reorient exactly from far endpoint to the observed camera-near opening.
        if float(np.dot(axis, opening_center - far_center)) < 0.0:
            axis = -axis
        pose_payload = {
            "ring_instance_id": int(ring.instance_id),
            "mouth_instance_id": int(mouth.instance_id),
            "normal_toward_camera": axis.tolist(),
            "opening_center_camera_mm": np.asarray(opening_center, dtype=np.float64).tolist(),
            "far_opening_center_camera_mm": np.asarray(far_center, dtype=np.float64).tolist(),
            "plane_offset": float(-np.dot(axis, opening_center)),
            "side_point_count": int(len(side_points)),
            "side_normal_count": int(len(normals)),
            "side_plane_inlier_ratio": float(best["radial_inlier_ratio"]),
            "side_residual_median_mm": float(best["radial_residual_median_mm"]),
            "side_residual_p95_mm": float(best["radial_residual_p90_mm"]),
            "diagnostics": {
                "opening_partial": True,
                "pose_source": "partial_mouth_plus_local_outer_cylinder",
                "association_mode": association_mode,
                "mouth_containment": float(containment),
                "mouth_to_ring_area_ratio": float(area_ratio),
                "side_point_count": int(len(side_points)),
                "side_normal_count": int(len(normals)),
                "side_component_count": int(component_count),
                "side_kept_component_count": int(kept_component_count),
                "radial_inlier_ratio": float(best["radial_inlier_ratio"]),
                "radial_residual_median_mm": float(best["radial_residual_median_mm"]),
                "radial_residual_p90_mm": float(best["radial_residual_p90_mm"]),
                "normal_axis_median_deg": float(best["normal_axis_median_deg"]),
                "normal_axis_p90_deg": float(best["normal_axis_p90_deg"]),
                "normal_radial_median_deg": float(best["normal_radial_median_deg"]),
                "normal_radial_p90_deg": float(best["normal_radial_p90_deg"]),
                "visible_normal_span_deg": float(best["visible_normal_span_deg"]),
                "projected_axis_error_deg": float(best["projected_axis_error_deg"]),
                "observed_axis_span_mm": float(best["observed_axis_span_mm"]),
                "endpoint_support_point_count": int(len(endpoint_points)),
                "endpoint_plane_inlier_count": int(endpoint_inlier_count),
                "endpoint_plane_inlier_ratio": float(endpoint_inlier_ratio),
                "endpoint_axial_residual_p90_mm": endpoint_axial_p90,
                "opening_center_error_px": mouth_center_error_px,
                "partial_mouth_overlap_ratio": mouth_overlap_ratio,
                "opening_near_margin_mm": opening_near_margin_mm,
                "axis_view_angle_deg": axis_view_angle_deg,
                "synthetic_mouth_area_px": int(np.count_nonzero(synthetic_mask)),
                "synthetic_outer_area_px": int(np.count_nonzero(synthetic_outer_mask)),
                "model_completed_opening": True,
                "candidate_axis_count": int(len(candidate_axes)),
                "fit_score": float(best["score"]),
            },
        }

    timing["total_ms"] = (time.perf_counter() - started) * 1000.0
    return {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "eligible": bool(eligible),
        "rejection_reasons": reasons,
        "warnings": warnings,
        "association": dict(association),
        "pose_payload": pose_payload,
        "synthetic_mouth_instance": synthetic_instance,
        "synthetic_ring_instance": synthetic_ring_instance,
        "diagnostics": (pose_payload or {}).get("diagnostics") or {
            "opening_partial": True,
            "association_mode": association_mode,
            "mouth_containment": float(containment),
            "mouth_to_ring_area_ratio": float(area_ratio),
            "side_point_count": int(len(side_points)),
            "side_normal_count": int(len(normals)),
            "endpoint_support_point_count": int(len(endpoint_points)),
            "endpoint_plane_inlier_count": int(endpoint_inlier_count),
            "endpoint_plane_inlier_ratio": float(endpoint_inlier_ratio),
            "candidate_axis_count": int(len(candidate_axes)),
            "radial_inlier_ratio": (best or {}).get("radial_inlier_ratio"),
            "radial_residual_median_mm": (best or {}).get("radial_residual_median_mm"),
            "radial_residual_p90_mm": (best or {}).get("radial_residual_p90_mm"),
            "normal_axis_median_deg": (best or {}).get("normal_axis_median_deg"),
            "normal_axis_p90_deg": (best or {}).get("normal_axis_p90_deg"),
            "visible_normal_span_deg": (best or {}).get("visible_normal_span_deg"),
            "projected_axis_error_deg": (best or {}).get("projected_axis_error_deg"),
            "observed_axis_span_mm": (best or {}).get("observed_axis_span_mm"),
            "normal_radial_median_deg": (best or {}).get("normal_radial_median_deg"),
            "normal_radial_p90_deg": (best or {}).get("normal_radial_p90_deg"),
            "endpoint_axial_residual_p90_mm": endpoint_axial_p90,
            "opening_center_error_px": mouth_center_error_px,
            "partial_mouth_overlap_ratio": mouth_overlap_ratio,
            "opening_near_margin_mm": opening_near_margin_mm,
            "axis_view_angle_deg": axis_view_angle_deg,
        },
        "timing_ms": timing,
        "_debug": {
            "side_surface_mask": side_mask,
            "depth_edge_mask": depth_edges,
            "endpoint_support_mask": endpoint_support_mask,
            "synthetic_mouth_mask": synthetic_mask,
            "synthetic_outer_mask": synthetic_outer_mask,
            "side_points_camera_mm": side_points,
            "side_pixels": side_pixels,
            "endpoint_points_camera_mm": endpoint_points,
            "endpoint_pixels": endpoint_pixels,
        },
    }
