"""M38.5 pure-side outer-contact geometry.

This branch deliberately does not infer a hidden opening and does not create an
inner-finger candidate.  It uses only the observed outer cylindrical side patch
to recover an *undirected* cylinder axis, a camera-near outer contact point and
the local inward closing direction.

The result is camera-frame geometry only and is explicitly not robot-ready.
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
    _deproject_mask,
    _depth_edge_mask,
    _dilate,
    _erode,
    _fit_circle_center_fixed_radius,
    _normal_covariance_axis,
    _organized_normals,
    _project_points,
    _retain_components,
    _safe_float,
    _safe_int,
    _unit,
)
from .partial_opening_cylinder_m383 import _surface_depth_mode

_EPS = 1e-9


def _axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = _unit(first)
    second = _unit(second)
    cosine = float(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _occupied_span_deg(angles: np.ndarray) -> float:
    values = np.mod(np.asarray(angles, dtype=np.float64), 2.0 * math.pi)
    if len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * math.pi)))
    return math.degrees(max(0.0, 2.0 * math.pi - float(np.max(gaps))))


def _canonical_undirected_axis(axis: np.ndarray) -> np.ndarray:
    result = _unit(axis)
    for value in result:
        if abs(float(value)) <= 1e-8:
            continue
        if float(value) < 0.0:
            result = -result
        break
    return result


def _local_axis_candidates(seed: np.ndarray, offsets_deg: Sequence[float]) -> List[np.ndarray]:
    seed = _unit(seed)
    basis_u, basis_v = _basis_perpendicular(seed)
    values = [float(value) for value in offsets_deg]
    if 0.0 not in values:
        values.append(0.0)
    candidates: List[np.ndarray] = []
    for first in values:
        for second in values:
            tangent_u = math.tan(math.radians(first))
            tangent_v = math.tan(math.radians(second))
            candidate = _unit(seed + tangent_u * basis_u + tangent_v * basis_v)
            if any(_axis_angle_deg(candidate, existing) < 1.0 for existing in candidates):
                continue
            candidates.append(candidate)
    return candidates


def _bootstrap_axis_stability(normals: np.ndarray, seed: np.ndarray, groups: int) -> Dict[str, Any]:
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    estimates: List[np.ndarray] = []
    group_count = max(2, int(groups))
    for group in range(group_count):
        subset = normals[group::group_count]
        if len(subset) < 20:
            continue
        estimate = _normal_covariance_axis(subset)
        if estimate is not None:
            estimates.append(estimate)
    errors = [_axis_angle_deg(seed, value) for value in estimates]
    pairwise: List[float] = []
    for first_index, first in enumerate(estimates):
        for second in estimates[first_index + 1 :]:
            pairwise.append(_axis_angle_deg(first, second))
    return {
        "sample_count": int(len(estimates)),
        "median_seed_error_deg": float(np.median(errors)) if errors else None,
        "maximum_seed_error_deg": float(max(errors)) if errors else None,
        "maximum_pairwise_dispersion_deg": float(max(pairwise)) if pairwise else 0.0,
    }


def _evaluate_axis(
    points: np.ndarray,
    normal_points: np.ndarray,
    normals: np.ndarray,
    axis: np.ndarray,
    outer_radius_mm: float,
    iterations: int,
    radial_inlier_threshold_mm: float,
) -> Dict[str, Any]:
    axis = _unit(axis)
    basis_u, basis_v = _basis_perpendicular(axis)
    projected = np.column_stack((points @ basis_u, points @ basis_v))
    center_2d = _fit_circle_center_fixed_radius(projected, outer_radius_mm, iterations)
    radial_2d = projected - center_2d[None, :]
    radial_distance = np.linalg.norm(radial_2d, axis=1)
    residual = np.abs(radial_distance - float(outer_radius_mm))
    inlier = residual <= float(radial_inlier_threshold_mm)
    radial_ratio = float(np.mean(inlier)) if len(inlier) else 0.0
    radial_median = float(np.median(residual)) if len(residual) else float("inf")
    radial_p90 = float(np.percentile(residual, 90)) if len(residual) else float("inf")

    axial = points @ axis
    span_values = axial[inlier] if int(np.count_nonzero(inlier)) >= 20 else axial
    observed_span = (
        float(np.percentile(span_values, 95) - np.percentile(span_values, 5))
        if len(span_values)
        else 0.0
    )
    median_axial = float(np.median(span_values)) if len(span_values) else float(np.median(axial))
    axis_point = basis_u * float(center_2d[0]) + basis_v * float(center_2d[1]) + axis * median_axial

    normal_axis_error = np.empty((0,), dtype=np.float64)
    normal_radial_error = np.empty((0,), dtype=np.float64)
    visible_span = 0.0
    if len(normals):
        normal_axis_error = np.degrees(np.arcsin(np.clip(np.abs(normals @ axis), 0.0, 1.0)))
        normal_projected = np.column_stack((normal_points @ basis_u, normal_points @ basis_v))
        normal_radial = normal_projected - center_2d[None, :]
        normal_radial_norm = np.linalg.norm(normal_radial, axis=1)
        valid = normal_radial_norm > 1e-6
        radial_unit = np.zeros_like(normal_radial)
        radial_unit[valid] = normal_radial[valid] / normal_radial_norm[valid, None]
        normal_2d = np.column_stack((normals @ basis_u, normals @ basis_v))
        normal_2d_norm = np.linalg.norm(normal_2d, axis=1)
        valid &= normal_2d_norm > 1e-6
        normalized_normals = np.zeros_like(normal_2d)
        normalized_normals[valid] = normal_2d[valid] / normal_2d_norm[valid, None]
        alignment = np.abs(np.sum(normalized_normals * radial_unit, axis=1))
        normal_radial_error = np.degrees(np.arccos(np.clip(alignment[valid], 0.0, 1.0)))
        visible_span = _occupied_span_deg(np.arctan2(normal_radial[valid, 1], normal_radial[valid, 0]))

    normal_axis_median = float(np.median(normal_axis_error)) if len(normal_axis_error) else 90.0
    normal_axis_p90 = float(np.percentile(normal_axis_error, 90)) if len(normal_axis_error) else 90.0
    normal_radial_median = float(np.median(normal_radial_error)) if len(normal_radial_error) else 90.0
    score = (
        radial_median
        + 0.25 * radial_p90
        + 8.0 * (1.0 - radial_ratio)
        + 0.05 * normal_axis_median
        + 0.025 * normal_radial_median
    )
    return {
        "score": float(score),
        "axis": axis,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "circle_center_2d": center_2d,
        "axis_point": axis_point,
        "radial_inlier_mask": inlier,
        "axial_coordinate_mm": axial,
        "radial_inlier_ratio": radial_ratio,
        "radial_residual_median_mm": radial_median,
        "radial_residual_p90_mm": radial_p90,
        "normal_axis_median_deg": normal_axis_median,
        "normal_axis_p90_deg": normal_axis_p90,
        "normal_radial_median_deg": normal_radial_median,
        "visible_normal_span_deg": float(visible_span),
        "observed_axis_span_mm": float(observed_span),
    }


def fit_side_surface_outer_contact_m385(
    ring: SegmentationInstance,
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recover an outer contact point from an observed cylindrical side patch."""

    started = time.perf_counter()
    section = raw_config.get("m38_branch_d") or {}
    if not isinstance(section, Mapping):
        section = {}
    depth_cfg = raw_config.get("depth") or {}
    if not isinstance(depth_cfg, Mapping):
        depth_cfg = {}
    object_cfg = raw_config.get("object_geometry") or {}
    if not isinstance(object_cfg, Mapping):
        object_cfg = {}

    reasons: List[str] = []
    warnings: List[str] = []
    timing: Dict[str, float] = {}
    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)

    prepare_started = time.perf_counter()
    mask = _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 2))
    other_mask = np.zeros_like(mask, dtype=bool)
    for other in all_rings:
        if int(other.instance_id) != int(ring.instance_id):
            other_mask |= other.mask.astype(bool)
    mask &= ~_dilate(other_mask, _safe_int(section.get("neighbor_exclusion_dilate_px"), 1))
    mask &= (depth_mm >= minimum_depth) & (depth_mm <= maximum_depth)
    surface_depth = _surface_depth_mode(depth_mm, mask)
    if surface_depth is None:
        reasons.append("m385_side_surface_depth_unavailable")
        surface_depth = float(np.median(depth_mm[mask])) if np.any(mask) else 0.0
    mask &= (
        (depth_mm >= float(surface_depth) - _safe_float(section.get("side_depth_front_tolerance_mm"), 16.0))
        & (depth_mm <= float(surface_depth) + _safe_float(section.get("side_depth_back_tolerance_mm"), 48.0))
    )
    depth_edges = _depth_edge_mask(
        depth_mm,
        mask,
        _safe_float(section.get("depth_edge_threshold_mm"), 18.0),
        _safe_int(section.get("depth_edge_dilate_px"), 1),
    )
    mask &= ~depth_edges
    mask, component_count, kept_component_count = _retain_components(
        mask, _safe_float(section.get("surface_component_minimum_ratio"), 0.10)
    )
    timing["side_mask_prepare_ms"] = (time.perf_counter() - prepare_started) * 1000.0

    extract_started = time.perf_counter()
    points, pixels = _deproject_mask(depth_mm, mask, intrinsics, minimum_depth, maximum_depth)
    normal_points, normals = _organized_normals(
        depth_mm, mask, intrinsics, _safe_int(section.get("normal_neighbor_step_px"), 2)
    )
    timing["point_and_normal_extract_ms"] = (time.perf_counter() - extract_started) * 1000.0
    if len(points) < _safe_int(section.get("minimum_side_points"), 100):
        reasons.append("m385_insufficient_side_surface_points")
    if len(normals) < _safe_int(section.get("minimum_normal_points"), 50):
        reasons.append("m385_insufficient_side_surface_normals")

    seed_started = time.perf_counter()
    seed = _normal_covariance_axis(normals)
    eigenvalue_ratio = None
    if len(normals) >= 3:
        covariance = normals.T @ normals / float(len(normals))
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
            eigenvalue_ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-9))
        except np.linalg.LinAlgError:
            eigenvalue_ratio = None
    if seed is None:
        reasons.append("m385_normal_covariance_axis_unavailable")
        seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    # Remove likely end faces after the first normal-axis estimate by keeping the
    # central axial interval. This preserves the curved crown while suppressing
    # flat end patches and their rounded transition.
    axial_all = points @ seed if len(points) else np.empty((0,), dtype=np.float64)
    central_fraction = min(0.95, max(0.30, _safe_float(section.get("central_axis_fraction"), 0.70)))
    tail = 0.5 * (1.0 - central_fraction)
    if len(axial_all) >= 30:
        low = float(np.quantile(axial_all, tail))
        high = float(np.quantile(axial_all, 1.0 - tail))
        central = (axial_all >= low) & (axial_all <= high)
        central_points = points[central]
        central_pixels = pixels[central]
    else:
        central_points = points
        central_pixels = pixels

    maximum_fit_points = max(100, _safe_int(section.get("maximum_fit_points"), 500))
    fit_points = central_points
    if len(fit_points) > maximum_fit_points:
        indexes = np.linspace(0, len(fit_points) - 1, maximum_fit_points).astype(np.int64)
        fit_points = fit_points[indexes]
    maximum_normal_points = max(50, _safe_int(section.get("maximum_normal_fit_points"), 400))
    fit_normal_points, fit_normals = normal_points, normals
    if len(fit_normals) > maximum_normal_points:
        indexes = np.linspace(0, len(fit_normals) - 1, maximum_normal_points).astype(np.int64)
        fit_normal_points = fit_normal_points[indexes]
        fit_normals = fit_normals[indexes]
    bootstrap = _bootstrap_axis_stability(
        fit_normals, seed, _safe_int(section.get("bootstrap_group_count"), 4)
    )
    timing["axis_seed_ms"] = (time.perf_counter() - seed_started) * 1000.0

    fit_started = time.perf_counter()
    offsets = section.get("local_axis_offsets_deg", [-12, -8, -4, 0, 4, 8, 12])
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
        offsets = [-12, -8, -4, 0, 4, 8, 12]
    candidates = _local_axis_candidates(seed, [float(value) for value in offsets])
    evaluations: List[Dict[str, Any]] = []
    if len(fit_points) >= 3:
        for axis in candidates:
            evaluations.append(
                _evaluate_axis(
                    fit_points,
                    fit_normal_points,
                    fit_normals,
                    axis,
                    outer_radius,
                    _safe_int(section.get("fixed_radius_iterations"), 4),
                    _safe_float(section.get("radial_inlier_threshold_mm"), 7.0),
                )
            )
    best = min(evaluations, key=lambda row: float(row["score"])) if evaluations else None
    timing["local_cylinder_fit_ms"] = (time.perf_counter() - fit_started) * 1000.0
    if best is None:
        reasons.append("m385_local_cylinder_fit_failed")

    contact = None
    contact_uv = None
    observed_contact = None
    support_error = None
    outer_normal = None
    closing_direction = None
    canonical_axis = None
    frame = None
    axis_view_angle_deg = None
    if best is not None:
        canonical_axis = _canonical_undirected_axis(np.asarray(best["axis"], dtype=np.float64))
        axis_point = np.asarray(best["axis_point"], dtype=np.float64)
        view_to_camera = _unit(-axis_point)
        axis_view_angle_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(float(np.dot(canonical_axis, view_to_camera))),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        to_camera = -axis_point
        radial = to_camera - canonical_axis * float(np.dot(to_camera, canonical_axis))
        try:
            outer_normal = _unit(radial)
        except ValueError:
            reasons.append("m385_camera_facing_radial_direction_unavailable")
        if outer_normal is not None:
            contact = axis_point + outer_radius * outer_normal
            closing_direction = -outer_normal
            projected = _project_points(contact[None, :], intrinsics)
            if np.isfinite(projected).all():
                contact_uv = projected[0]
            else:
                reasons.append("m385_outer_contact_not_projectable")

            inlier = np.asarray(best["radial_inlier_mask"], dtype=bool)
            axial = np.asarray(best["axial_coordinate_mm"], dtype=np.float64)
            center_axial = float(np.dot(axis_point, canonical_axis))
            support_mask = inlier & (
                np.abs(axial - center_axial)
                <= _safe_float(section.get("contact_support_axial_half_span_mm"), 12.0)
            )
            support_points = fit_points[support_mask]
            if len(support_points):
                distances = np.linalg.norm(support_points - contact[None, :], axis=1)
                best_indexes = np.argsort(distances)[: max(1, min(8, len(distances)))]
                observed_contact = np.median(support_points[best_indexes], axis=0)
                support_error = float(np.linalg.norm(observed_contact - contact))
            else:
                reasons.append("m385_outer_contact_lacks_observed_support")

            frame_z = np.cross(closing_direction, canonical_axis)
            try:
                frame_z = _unit(frame_z)
                frame_y = _unit(np.cross(frame_z, closing_direction))
                frame = {
                    "x_closing_direction": closing_direction.tolist(),
                    "y_cylinder_axis": frame_y.tolist(),
                    "z_tangent_direction": frame_z.tolist(),
                }
            except ValueError:
                reasons.append("m385_outer_contact_frame_unavailable")

        if best["radial_inlier_ratio"] < _safe_float(section.get("minimum_radial_inlier_ratio"), 0.65):
            reasons.append("m385_local_cylinder_inlier_ratio_too_low")
        if best["radial_residual_median_mm"] > _safe_float(section.get("maximum_radial_residual_median_mm"), 5.0):
            reasons.append("m385_local_cylinder_residual_median_too_high")
        if best["radial_residual_p90_mm"] > _safe_float(section.get("maximum_radial_residual_p90_mm"), 12.0):
            reasons.append("m385_local_cylinder_residual_p90_too_high")
        if best["normal_axis_median_deg"] > _safe_float(section.get("maximum_normal_axis_median_deg"), 24.0):
            reasons.append("m385_side_normals_not_perpendicular_to_axis")
        if best["normal_axis_p90_deg"] > _safe_float(section.get("maximum_normal_axis_p90_deg"), 52.0):
            reasons.append("m385_side_normal_axis_p90_too_high")
        if best["normal_radial_median_deg"] > _safe_float(section.get("maximum_normal_radial_median_deg"), 35.0):
            reasons.append("m385_side_normals_not_radial")
        if best["visible_normal_span_deg"] < _safe_float(section.get("minimum_visible_normal_span_deg"), 60.0):
            reasons.append("m385_visible_cylinder_arc_too_small")
        if best["observed_axis_span_mm"] < _safe_float(section.get("minimum_observed_axis_span_mm"), 25.0):
            reasons.append("m385_observed_axis_span_too_short")
        if axis_view_angle_deg < _safe_float(section.get("minimum_axis_view_angle_deg"), 65.0):
            reasons.append("m385_axis_not_side_on_enough")

    if eigenvalue_ratio is not None and eigenvalue_ratio > _safe_float(
        section.get("maximum_normal_axis_eigenvalue_ratio"), 0.55
    ):
        reasons.append("m385_normal_axis_not_well_constrained")
    maximum_dispersion = bootstrap.get("maximum_pairwise_dispersion_deg")
    if maximum_dispersion is not None and float(maximum_dispersion) > _safe_float(
        section.get("maximum_bootstrap_axis_dispersion_deg"), 18.0
    ):
        reasons.append("m385_bootstrap_axis_unstable")
    if support_error is not None and support_error > _safe_float(
        section.get("maximum_contact_support_error_mm"), 14.0
    ):
        reasons.append("m385_outer_contact_not_supported_by_observed_surface")
    if contact_uv is not None:
        x = int(round(float(contact_uv[0])))
        y = int(round(float(contact_uv[1])))
        if not (0 <= y < ring.mask.shape[0] and 0 <= x < ring.mask.shape[1] and bool(ring.mask[y, x])):
            reasons.append("m385_outer_contact_projects_outside_ring_mask")
        else:
            distance = cv2.distanceTransform(ring.mask.astype(np.uint8), cv2.DIST_L2, 5)
            if float(distance[y, x]) < _safe_float(section.get("minimum_contact_inside_mask_px"), 2.0):
                warnings.append("m385_outer_contact_near_segmentation_boundary")

    eligible = bool(
        not reasons
        and best is not None
        and contact is not None
        and contact_uv is not None
        and outer_normal is not None
        and closing_direction is not None
        and canonical_axis is not None
    )
    diagnostics = {
        "surface_depth_mm": float(surface_depth),
        "side_point_count": int(len(points)),
        "central_side_point_count": int(len(central_points)),
        "side_normal_count": int(len(normals)),
        "side_component_count": int(component_count),
        "side_kept_component_count": int(kept_component_count),
        "candidate_axis_count": int(len(evaluations)),
        "normal_axis_eigenvalue_ratio": eigenvalue_ratio,
        "bootstrap": bootstrap,
        "radial_inlier_ratio": (best or {}).get("radial_inlier_ratio"),
        "radial_residual_median_mm": (best or {}).get("radial_residual_median_mm"),
        "radial_residual_p90_mm": (best or {}).get("radial_residual_p90_mm"),
        "normal_axis_median_deg": (best or {}).get("normal_axis_median_deg"),
        "normal_axis_p90_deg": (best or {}).get("normal_axis_p90_deg"),
        "normal_radial_median_deg": (best or {}).get("normal_radial_median_deg"),
        "visible_normal_span_deg": (best or {}).get("visible_normal_span_deg"),
        "observed_axis_span_mm": (best or {}).get("observed_axis_span_mm"),
        "axis_view_angle_deg": axis_view_angle_deg,
        "contact_support_error_mm": support_error,
        "axis_direction_ambiguous": True,
    }
    candidate = None
    if eligible:
        candidate = {
            "schema_version": "1.0",
            "candidate_type": "outer_contact_geometry",
            "grasp_branch": "m38_5_side_surface_outer_contact_only",
            "grasp_mode": "outer_contact_only",
            "pose_source": "observed_outer_cylinder_side_surface",
            "robot_ready": False,
            "robot_ready_reason": (
                "outer contact geometry only; inner-finger entry, complete gripper collision and robot reachability are intentionally not evaluated"
            ),
            "target": {
                "ring_instance_id": int(ring.instance_id),
                "ring_confidence": float(ring.confidence),
                "surface_depth_mm": float(surface_depth),
            },
            "outer_contact": {
                "contact_camera_mm": np.asarray(contact, dtype=np.float64).tolist(),
                "contact_uv": np.asarray(contact_uv, dtype=np.float64).tolist(),
                "observed_support_camera_mm": (
                    np.asarray(observed_contact, dtype=np.float64).tolist()
                    if observed_contact is not None else None
                ),
                "support_error_mm": support_error,
                "outer_surface_normal_camera": np.asarray(outer_normal, dtype=np.float64).tolist(),
                "closing_direction_camera": np.asarray(closing_direction, dtype=np.float64).tolist(),
                "cylinder_axis_camera_undirected": np.asarray(canonical_axis, dtype=np.float64).tolist(),
                "axis_direction_ambiguous": True,
                "contact_frame_camera": frame,
            },
            "quality": diagnostics,
            "warnings": warnings,
        }

    timing["total_ms"] = (time.perf_counter() - started) * 1000.0
    return {
        "ring_instance_id": int(ring.instance_id),
        "eligible": bool(eligible),
        "rejection_reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "candidate": candidate,
        "diagnostics": diagnostics,
        "timing_ms": timing,
        "_debug": {
            "side_surface_mask": mask,
            "depth_edge_mask": depth_edges,
            "side_points_camera_mm": points,
            "side_pixels": pixels,
            "central_side_pixels": central_pixels,
        },
    }
