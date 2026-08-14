"""M39.4.1 camera-facing arc opening reconstruction for pure side-lying rings.

This stage consumes the reliable undirected cylinder axis from M39.4.0.1 and
reconstructs geometry from the target ring itself.  It deliberately does NOT
assume that the ring rests on the box floor: stacked/elevated rings remain
valid as long as the camera-facing outer cylinder arc is observable.

Outputs are validation-only:
- fixed-radius camera-facing outer-arc cross-section centre line;
- selected-end opening plane / opening centre from axial support drop;
- side grasp frame using the existing Visual Grasp Frame contract: +Z is axial
  insertion/approach, +X points from the hole toward the observed camera-facing
  outer wall (closing), and +Y is the lateral axis;
- a preview grasp origin a short distance inside the opening.

No robot candidate is created here.  Collision-aware routing and robot motion
remain disabled until the next production-enablement step.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .partial_opening_cylinder import (
    _deproject_mask,
    _depth_edge_mask,
    _dilate,
    _erode,
    _safe_float,
    _safe_int,
    _unit,
)
from .robot_pose_transform import _rotation_to_quaternion_xyzw
from .segmentation import SegmentationInstance

_EPS = 1e-9


def _json_vector(value: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def _project(point: np.ndarray, intrinsics: Mapping[str, float]) -> np.ndarray:
    x, y, z = [float(v) for v in point]
    if z <= 1e-6:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)
    return np.asarray([
        float(intrinsics["fx"]) * x / z + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * y / z + float(intrinsics["cy"]),
    ], dtype=np.float64)


def _robust_fixed_radius_center(
    points_2d: np.ndarray,
    radius_mm: float,
    *,
    huber_delta_mm: float,
    maximum_iterations: int,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Fit a known-radius circle centre to a visible camera-facing arc."""

    if len(points_2d) < 8:
        raise ValueError("insufficient arc points")
    # In the local basis +u points from the cylinder centre toward the camera.
    # Therefore camera-facing samples should sit roughly +R from the centre.
    centre = np.median(points_2d, axis=0) - np.asarray([radius_mm, 0.0])
    for _iteration in range(max(1, int(maximum_iterations))):
        delta = points_2d - centre
        distance = np.linalg.norm(delta, axis=1)
        distance = np.maximum(distance, 1e-6)
        residual = distance - radius_mm
        threshold = max(0.5, float(huber_delta_mm))
        weight = np.ones_like(residual)
        mask = np.abs(residual) > threshold
        weight[mask] = threshold / np.maximum(np.abs(residual[mask]), 1e-9)
        jacobian = -delta / distance[:, None]
        normal = jacobian.T @ (weight[:, None] * jacobian) + 1e-6 * np.eye(2)
        rhs = -(jacobian.T @ (weight * residual))
        step = np.linalg.solve(normal, rhs)
        centre = centre + step
        if float(np.linalg.norm(step)) < 1e-4:
            break

    residual_abs = np.abs(np.linalg.norm(points_2d - centre, axis=1) - radius_mm)
    return centre, {
        "residual_median_mm": float(np.median(residual_abs)),
        "residual_p90_mm": float(np.percentile(residual_abs, 90)),
    }, residual_abs


def _angular_span_deg(radial_2d: np.ndarray) -> float:
    if len(radial_2d) < 3:
        return 0.0
    angle = np.degrees(np.arctan2(radial_2d[:, 1], radial_2d[:, 0]))
    # Camera-facing arc is deliberately kept around +u, so wrapping at +/-180
    # is not expected. Percentile span suppresses isolated segmentation points.
    return float(np.percentile(angle, 95) - np.percentile(angle, 5))


def _front_envelope_points(
    points: np.ndarray,
    axis: np.ndarray,
    *,
    central_fraction: float,
    axial_bins: int,
    front_quantile: float,
    front_band_mm: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep the locally nearest depth layer over the central cylinder length."""

    s = points @ axis
    trim = 0.5 * (1.0 - float(np.clip(central_fraction, 0.20, 0.95)))
    low = float(np.quantile(s, trim))
    high = float(np.quantile(s, 1.0 - trim))
    central = (s >= low) & (s <= high)
    central_points = points[central]
    central_s = s[central]
    if len(central_points) < 8:
        return central_points, {
            "central_s_min_mm": low,
            "central_s_max_mm": high,
            "central_point_count": int(len(central_points)),
            "front_point_count": int(len(central_points)),
        }

    keep = np.zeros(len(central_points), dtype=bool)
    edges = np.linspace(low, high, max(4, int(axial_bins)) + 1)
    q = float(np.clip(front_quantile, 0.02, 0.45))
    band = max(1.0, float(front_band_mm))
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            local = np.where((central_s >= edges[index]) & (central_s <= edges[index + 1]))[0]
        else:
            local = np.where((central_s >= edges[index]) & (central_s < edges[index + 1]))[0]
        if len(local) < 4:
            continue
        local_depth = central_points[local, 2]
        front = float(np.quantile(local_depth, q))
        keep[local[local_depth <= front + band]] = True
    return central_points[keep], {
        "central_s_min_mm": low,
        "central_s_max_mm": high,
        "central_point_count": int(len(central_points)),
        "front_point_count": int(np.count_nonzero(keep)),
    }


def _opening_support_drop(
    shell_s: np.ndarray,
    *,
    outward_sign: float,
    bin_mm: float,
    inward_reference_min_mm: float,
    inward_reference_max_mm: float,
    transition_window_mm: float,
    maximum_drop_ratio: float,
    minimum_reference_count: float,
) -> Dict[str, Any]:
    """Locate the selected cylinder end from the outward axial support drop."""

    if len(shell_s) < 20:
        return {"status": "insufficient_shell_support"}
    u = outward_sign * np.asarray(shell_s, dtype=np.float64)
    robust_outer = float(np.percentile(u, 99.0))
    step = max(1.0, float(bin_mm))
    start = robust_outer - max(35.0, float(inward_reference_max_mm) + 10.0)
    stop = robust_outer + max(8.0, 0.5 * float(transition_window_mm))
    edges = np.arange(start, stop + step, step)
    if len(edges) < 8:
        return {"status": "invalid_support_profile"}
    counts, _ = np.histogram(u, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if len(counts) >= 3:
        smooth = np.convolve(counts.astype(np.float64), np.asarray([0.25, 0.50, 0.25]), mode="same")
    else:
        smooth = counts.astype(np.float64)

    ref_lo = robust_outer - float(inward_reference_max_mm)
    ref_hi = robust_outer - float(inward_reference_min_mm)
    ref_mask = (centres >= ref_lo) & (centres <= ref_hi)
    reference = float(np.median(smooth[ref_mask])) if np.any(ref_mask) else 0.0
    if reference < float(minimum_reference_count):
        return {
            "status": "weak_interior_support",
            "robust_outer_u_mm": robust_outer,
            "reference_support": reference,
            "profile_centres_u_mm": [float(v) for v in centres],
            "profile_counts": [int(v) for v in counts],
            "profile_smooth": [float(v) for v in smooth],
        }

    window = max(6.0, float(transition_window_mm))
    candidate_indexes = np.where((centres >= robust_outer - window) & (centres <= robust_outer + 0.5 * step))[0]
    best = None
    for index in candidate_indexes:
        if index < 2 or index + 1 >= len(smooth):
            continue
        inner = float(np.median(smooth[max(0, index - 2): index + 1]))
        outer = float(np.median(smooth[index + 1: min(len(smooth), index + 3)]))
        drop = inner - outer
        ratio = outer / max(inner, 1e-6)
        score = drop / max(reference, 1e-6)
        if best is None or score > best["score"]:
            best = {
                "index": int(index),
                "inner_support": inner,
                "outer_support": outer,
                "drop_ratio": ratio,
                "score": score,
            }
    if best is None:
        return {"status": "transition_not_found", "robust_outer_u_mm": robust_outer}

    index = int(best["index"])
    opening_u = float(edges[index + 1])
    status = "support_drop_found" if float(best["drop_ratio"]) <= float(maximum_drop_ratio) else "weak_support_drop"
    return {
        "status": status,
        "opening_u_mm": opening_u,
        "robust_outer_u_mm": robust_outer,
        "reference_support": reference,
        "inner_support": float(best["inner_support"]),
        "outer_support": float(best["outer_support"]),
        "drop_ratio": float(best["drop_ratio"]),
        "drop_score": float(best["score"]),
        "profile_centres_u_mm": [float(v) for v in centres],
        "profile_counts": [int(v) for v in counts],
        "profile_smooth": [float(v) for v in smooth],
    }


def reconstruct_side_opening(
    ring: SegmentationInstance,
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    axis_fit: Mapping[str, Any],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    section = raw_config.get("m39_4_1_side_opening_reconstruction") or {}
    depth_cfg = raw_config.get("depth") or {}
    object_cfg = raw_config.get("object_geometry") or {}
    if not isinstance(section, Mapping):
        section = {}
    if not isinstance(depth_cfg, Mapping):
        depth_cfg = {}
    if not isinstance(object_cfg, Mapping):
        object_cfg = {}

    result: Dict[str, Any] = {
        "ring_instance_id": int(ring.instance_id),
        "status": "opening_uncertain",
        "reliable": False,
        "rejection_reasons": [],
        "warnings": [],
        "robot_ready": False,
        "robot_routing_enabled": False,
    }
    reasons: List[str] = result["rejection_reasons"]
    warnings: List[str] = result["warnings"]

    if not bool(axis_fit.get("axis_reliable", axis_fit.get("reliable", False))):
        reasons.append("m3941_source_axis_not_reliable")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result
    axis_raw = axis_fit.get("axis_camera_undirected")
    if not (isinstance(axis_raw, list) and len(axis_raw) == 3):
        reasons.append("m3941_source_axis_missing")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result
    axis = _unit(np.asarray(axis_raw, dtype=np.float64))
    entry_label = str(axis_fit.get("entry_endpoint") or "")
    if entry_label not in {"A", "B"}:
        reasons.append("m3941_entry_endpoint_missing")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result
    outward_sign = 1.0 if entry_label == "B" else -1.0
    outward_axis = outward_sign * axis
    inward_axis = -outward_axis

    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)
    inner_radius = 0.5 * _safe_float(object_cfg.get("nominal_inner_diameter_mm"), 60.0)
    axial_length = _safe_float(object_cfg.get("axial_length_mm"), 70.0)

    mask = _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 1))
    other = np.zeros_like(mask, dtype=bool)
    for item in all_rings:
        if int(item.instance_id) != int(ring.instance_id):
            other |= item.mask.astype(bool)
    mask &= ~_dilate(other, _safe_int(section.get("neighbor_exclusion_dilate_px"), 1))
    mask &= (depth_mm >= minimum_depth) & (depth_mm <= maximum_depth)
    edge = _depth_edge_mask(
        depth_mm,
        mask,
        _safe_float(section.get("depth_edge_threshold_mm"), 20.0),
        _safe_int(section.get("depth_edge_dilate_px"), 1),
    )
    mask &= ~edge
    points, _pixels = _deproject_mask(depth_mm, mask, intrinsics, minimum_depth, maximum_depth)
    result["surface_point_count"] = int(len(points))
    if len(points) < _safe_int(section.get("minimum_surface_points"), 180):
        reasons.append("m3941_insufficient_surface_points")
    if len(points) < 8:
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result

    anchor = np.median(points, axis=0)
    camera_radial = -anchor
    camera_radial = camera_radial - axis * float(np.dot(camera_radial, axis))
    if float(np.linalg.norm(camera_radial)) <= 1e-6:
        reasons.append("m3941_camera_facing_radial_degenerate")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result
    u_axis = _unit(camera_radial)
    tangent = _unit(np.cross(axis, u_axis))

    arc_points, envelope = _front_envelope_points(
        points,
        axis,
        central_fraction=_safe_float(section.get("central_axis_fraction"), 0.60),
        axial_bins=_safe_int(section.get("front_envelope_axial_bins"), 12),
        front_quantile=_safe_float(section.get("front_envelope_quantile"), 0.20),
        front_band_mm=_safe_float(section.get("front_envelope_band_mm"), 12.0),
    )
    result["front_envelope"] = envelope
    if len(arc_points) < _safe_int(section.get("minimum_arc_points"), 120):
        reasons.append("m3941_insufficient_camera_facing_arc_points")
    if len(arc_points) < 8:
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result

    arc_2d = np.column_stack((arc_points @ u_axis, arc_points @ tangent))
    try:
        initial_centre_2d, initial_quality, initial_residual_abs = _robust_fixed_radius_center(
            arc_2d,
            outer_radius,
            huber_delta_mm=_safe_float(section.get("arc_fit_huber_delta_mm"), 4.0),
            maximum_iterations=_safe_int(section.get("arc_fit_iterations"), 12),
        )
        refit_gate = _safe_float(section.get("arc_refit_gate_mm"), 8.0)
        refit_mask = initial_residual_abs <= refit_gate
        minimum_refit_points = _safe_int(section.get("minimum_arc_refit_points"), 100)
        if int(np.count_nonzero(refit_mask)) >= minimum_refit_points:
            centre_2d, fit_quality, refit_residual_abs = _robust_fixed_radius_center(
                arc_2d[refit_mask],
                outer_radius,
                huber_delta_mm=_safe_float(section.get("arc_fit_huber_delta_mm"), 4.0),
                maximum_iterations=_safe_int(section.get("arc_fit_iterations"), 12),
            )
            fit_points_2d = arc_2d[refit_mask]
            residual_abs = refit_residual_abs
        else:
            centre_2d = initial_centre_2d
            fit_quality = initial_quality
            fit_points_2d = arc_2d
            residual_abs = initial_residual_abs
            refit_mask = np.ones(len(arc_2d), dtype=bool)
            warnings.append("m3941_arc_refit_gate_had_too_few_points")
    except Exception as error:
        reasons.append("m3941_fixed_radius_arc_fit_failed")
        result["arc_fit_error"] = str(error)
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result

    inlier_threshold = _safe_float(section.get("arc_inlier_threshold_mm"), 5.0)
    inlier = residual_abs <= inlier_threshold
    inlier_ratio = float(np.mean(inlier)) if len(inlier) else 0.0
    raw_final_residual = np.abs(np.linalg.norm(arc_2d - centre_2d, axis=1) - outer_radius)
    raw_inlier_ratio = float(np.mean(raw_final_residual <= inlier_threshold)) if len(raw_final_residual) else 0.0
    radial_2d = fit_points_2d[inlier] - centre_2d if np.any(inlier) else fit_points_2d - centre_2d
    radial_norm = np.linalg.norm(radial_2d, axis=1)
    valid_radial = radial_norm > 1e-6
    radial_unit_2d = radial_2d[valid_radial] / radial_norm[valid_radial, None]
    minimum_camera_facing_component = _safe_float(section.get("minimum_camera_facing_radial_component"), 0.05)
    camera_side = radial_unit_2d[:, 0] >= minimum_camera_facing_component if len(radial_unit_2d) else np.zeros(0, dtype=bool)
    camera_radial_2d = radial_2d[valid_radial][camera_side] if len(radial_unit_2d) else np.empty((0, 2))
    camera_radial_unit_2d = radial_unit_2d[camera_side] if len(radial_unit_2d) else np.empty((0, 2))
    arc_span = _angular_span_deg(camera_radial_2d)
    if len(camera_radial_unit_2d) < 3:
        reasons.append("m3941_arc_radial_direction_unavailable")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result
    robust_radial_2d = np.median(camera_radial_unit_2d, axis=0)
    robust_radial_2d = robust_radial_2d / max(float(np.linalg.norm(robust_radial_2d)), 1e-9)
    radial_camera = _unit(robust_radial_2d[0] * u_axis + robust_radial_2d[1] * tangent)
    # It must point from the centre toward the camera-facing wall, not toward
    # the hidden/back half of the cylinder.
    if float(np.dot(radial_camera, u_axis)) < 0.0:
        radial_camera = -radial_camera
        warnings.append("m3941_camera_facing_radial_flipped_to_camera_side")

    centre_line_point = centre_2d[0] * u_axis + centre_2d[1] * tangent
    all_s = points @ axis
    all_radial = points - centre_line_point - all_s[:, None] * axis
    all_rho = np.linalg.norm(all_radial, axis=1)
    shell_tolerance = _safe_float(section.get("opening_shell_tolerance_mm"), 6.0)
    shell = np.abs(all_rho - outer_radius) <= shell_tolerance
    shell_s = all_s[shell]
    support = _opening_support_drop(
        shell_s,
        outward_sign=outward_sign,
        bin_mm=_safe_float(section.get("opening_support_bin_mm"), 2.0),
        inward_reference_min_mm=_safe_float(section.get("opening_reference_inward_min_mm"), 10.0),
        inward_reference_max_mm=_safe_float(section.get("opening_reference_inward_max_mm"), 28.0),
        transition_window_mm=_safe_float(section.get("opening_transition_window_mm"), 14.0),
        maximum_drop_ratio=_safe_float(section.get("maximum_opening_drop_ratio"), 0.50),
        minimum_reference_count=_safe_float(section.get("minimum_opening_reference_support"), 8.0),
    )
    result["opening_support"] = support
    opening_status = str(support.get("status") or "")
    if opening_status != "support_drop_found":
        reasons.append("m3941_opening_plane_support_drop_uncertain")
        # Still expose a diagnostic fallback at the robust shell extreme.
        if support.get("robust_outer_u_mm") is None:
            result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
            return result
        opening_u = float(support["robust_outer_u_mm"])
        warnings.append("m3941_opening_plane_using_outer_shell_fallback")
    else:
        opening_u = float(support["opening_u_mm"])

    opening_s = outward_sign * opening_u
    opening_center = centre_line_point + opening_s * axis
    opening_uv = _project(opening_center, intrinsics)

    # Preserve the existing M38.6/M39.x Visual Grasp Frame contract:
    #   +X = closing (inner-finger side -> outer-finger side)
    #   +Z = approach/insertion (TCP +X after T_grasp_hand_tcp)
    frame_z = _unit(inward_axis)
    frame_x = radial_camera - frame_z * float(np.dot(radial_camera, frame_z))
    frame_x = _unit(frame_x)
    frame_y = _unit(np.cross(frame_z, frame_x))
    frame_x = _unit(np.cross(frame_y, frame_z))
    rotation = np.column_stack((frame_x, frame_y, frame_z))
    quaternion = _rotation_to_quaternion_xyzw(rotation)

    insertion_depth = _safe_float(section.get("preview_insertion_depth_mm"), 18.0)
    preview_grasp_center = opening_center + insertion_depth * frame_z
    preview_grasp_uv = _project(preview_grasp_center, intrinsics)
    frame_axis_draw_mm = _safe_float(section.get("frame_axis_draw_length_mm"), 30.0)
    frame_x_tip_uv = _project(opening_center + frame_axis_draw_mm * frame_x, intrinsics)
    frame_z_tip_uv = _project(opening_center + frame_axis_draw_mm * frame_z, intrinsics)

    nominal_endpoint = axis_fit.get("entry_center_camera_mm")
    nominal_shift = None
    if isinstance(nominal_endpoint, list) and len(nominal_endpoint) == 3:
        nominal_s = float(np.dot(np.asarray(nominal_endpoint, dtype=np.float64), axis))
        nominal_shift = float(opening_s - nominal_s)

    arc_fit = {
        "fixed_outer_radius_mm": float(outer_radius),
        "centre_line_point_camera_mm": _json_vector(centre_line_point),
        "centre_2d_camera_facing_tangent_mm": _json_vector(centre_2d),
        "camera_facing_basis_u_camera": _json_vector(u_axis),
        "cross_section_tangent_camera": _json_vector(tangent),
        "arc_point_count": int(len(arc_points)),
        "arc_refit_point_count": int(len(fit_points_2d)),
        "arc_refit_ratio": float(len(fit_points_2d) / max(len(arc_points), 1)),
        "arc_inlier_count": int(np.count_nonzero(inlier)),
        "arc_inlier_ratio": float(inlier_ratio),
        "raw_arc_inlier_ratio": float(raw_inlier_ratio),
        "camera_facing_arc_point_count": int(len(camera_radial_unit_2d)),
        "minimum_camera_facing_radial_component": float(minimum_camera_facing_component),
        "arc_span_deg_p5_p95": float(arc_span),
        "residual_median_mm": float(fit_quality["residual_median_mm"]),
        "residual_p90_mm": float(fit_quality["residual_p90_mm"]),
        "measured_camera_facing_radial_camera": _json_vector(radial_camera),
    }
    result["camera_facing_outer_arc"] = arc_fit

    if float(fit_quality["residual_median_mm"]) > _safe_float(section.get("maximum_arc_residual_median_mm"), 3.0):
        reasons.append("m3941_arc_residual_median_too_large")
    if float(fit_quality["residual_p90_mm"]) > _safe_float(section.get("maximum_arc_residual_p90_mm"), 6.0):
        reasons.append("m3941_arc_residual_p90_too_large")
    if int(np.count_nonzero(inlier)) < _safe_int(section.get("minimum_arc_inlier_count"), 120):
        reasons.append("m3941_arc_inlier_count_too_small")
    if int(len(camera_radial_unit_2d)) < _safe_int(section.get("minimum_camera_facing_arc_points"), 100):
        reasons.append("m3941_camera_facing_arc_support_too_small")
    if inlier_ratio < _safe_float(section.get("minimum_arc_inlier_ratio"), 0.80):
        reasons.append("m3941_arc_inlier_ratio_too_small")
    if raw_inlier_ratio < _safe_float(section.get("minimum_raw_arc_inlier_ratio"), 0.65):
        reasons.append("m3941_raw_arc_support_too_small")
    elif raw_inlier_ratio < 0.80:
        warnings.append("m3941_arc_contains_substantial_non_cylinder_contamination")
    if arc_span < _safe_float(section.get("minimum_arc_span_deg"), 45.0):
        reasons.append("m3941_arc_span_too_small")

    reliable = len(reasons) == 0
    result.update({
        "status": "opening_frame_reconstructed" if reliable else "opening_uncertain",
        "reliable": bool(reliable),
        "axis_source_stage": "M39.4.0.1",
        "entry_endpoint": entry_label,
        "entry_selection_rule": axis_fit.get("entry_selection_rule"),
        "axis_camera_undirected": _json_vector(axis),
        "axis_image_angle_deg_0_180": axis_fit.get("axis_image_angle_deg_0_180"),
        "outward_axis_camera": _json_vector(outward_axis),
        "insertion_axis_camera": _json_vector(frame_z),
        "closing_axis_camera": _json_vector(frame_x),
        "nominal_outer_radius_mm": float(outer_radius),
        "nominal_inner_radius_mm": float(inner_radius),
        "nominal_axial_length_mm": float(axial_length),
        "opening_plane_s_camera_axis_mm": float(opening_s),
        "opening_center_camera_mm": _json_vector(opening_center),
        "opening_center_uv": _json_vector(opening_uv),
        "opening_shift_vs_m39401_nominal_endpoint_mm": nominal_shift,
        "preview_insertion_depth_mm": float(insertion_depth),
        "preview_grasp_center_camera_mm": _json_vector(preview_grasp_center),
        "preview_grasp_center_uv": _json_vector(preview_grasp_uv),
        "frame_axis_draw_length_mm": float(frame_axis_draw_mm),
        "frame_x_tip_uv": _json_vector(frame_x_tip_uv),
        "frame_z_tip_uv": _json_vector(frame_z_tip_uv),
        "opening_frame_camera": {
            "origin_camera_mm": _json_vector(opening_center),
            "coordinate_contract": "m38_6_visual_grasp",
            "x_closing_axis_camera": _json_vector(frame_x),
            "y_lateral_axis_camera": _json_vector(frame_y),
            "z_approach_axis_camera": _json_vector(frame_z),
            "tcp_forward_insertion_axis_camera": _json_vector(frame_z),
            "inner_to_outer_closing_axis_camera": _json_vector(frame_x),
            "rotation_matrix_rows": [[float(v) for v in row] for row in rotation],
            "quaternion_xyzw": _json_vector(quaternion),
            "inner_finger_side": "negative_x",
            "outer_finger_side": "positive_x",
        },
        "side_grasp_frame_camera": {
            "origin_camera_mm": _json_vector(preview_grasp_center),
            "origin_policy": "opening_center_plus_preview_insertion_depth_along_visual_plus_z",
            "coordinate_contract": "m38_6_visual_grasp",
            "x_closing_axis_camera": _json_vector(frame_x),
            "y_lateral_axis_camera": _json_vector(frame_y),
            "z_approach_axis_camera": _json_vector(frame_z),
            "tcp_forward_insertion_axis_camera": _json_vector(frame_z),
            "inner_to_outer_closing_axis_camera": _json_vector(frame_x),
            "rotation_matrix_rows": [[float(v) for v in row] for row in rotation],
            "quaternion_xyzw": _json_vector(quaternion),
            "inner_finger_side": "negative_x",
            "outer_finger_side": "positive_x",
        },
        "robot_ready": False,
        "robot_routing_enabled": False,
    })
    result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
    return result


def attach_m3941_side_opening_reconstruction(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    section = raw_config.get("m39_4_1_side_opening_reconstruction") or {}
    if not isinstance(section, Mapping):
        section = {}
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.4.1_camera_facing_arc_opening_reconstruction",
        "enabled": bool(section.get("enabled", True)),
        "mode": str(section.get("mode") or "online_validation_only"),
        "robot_routing_enabled": False,
        "robot_ready": False,
        "executed": False,
        "status": "disabled" if not bool(section.get("enabled", True)) else "not_applicable",
        "fits": [],
    }
    if not summary["enabled"]:
        scene["m39_4_1_side_opening_reconstruction"] = summary
        return summary

    source = scene.get("m39_4_0_side_axis_recovery")
    selected_axis = source.get("selected") if isinstance(source, Mapping) and isinstance(source.get("selected"), Mapping) else None
    if not isinstance(selected_axis, Mapping) or not bool(selected_axis.get("axis_reliable", selected_axis.get("reliable", False))):
        summary["status"] = "no_reliable_m39401_side_axis"
        scene["m39_4_1_side_opening_reconstruction"] = summary
        return summary

    ring_id = int(selected_axis.get("ring_instance_id"))
    all_rings = [item for item in instances if item.class_name == "foam_ring"]
    ring = next((item for item in all_rings if int(item.instance_id) == ring_id), None)
    if ring is None:
        summary["status"] = "selected_ring_instance_missing"
        scene["m39_4_1_side_opening_reconstruction"] = summary
        return summary

    fit = reconstruct_side_opening(ring, all_rings, depth_mm, intrinsics, selected_axis, raw_config)
    summary.update({
        "executed": True,
        "fits": [fit],
        "selected_ring_instance_id": ring_id,
        "selected": fit if bool(fit.get("reliable", False)) else None,
        "diagnostic": fit,
        "status": "opening_frame_reconstructed_validation_only" if bool(fit.get("reliable", False)) else "opening_reconstruction_uncertain",
        "selected_grasp_branch": "m39_4_1_side_opening_frame_validation" if bool(fit.get("reliable", False)) else "m39_4_1_side_opening_uncertain",
        "terminal_reject": True,
        "reason": "m3941_opening_frame_validation_only" if bool(fit.get("reliable", False)) else "m3941_opening_reconstruction_uncertain",
        "display_reason_short": "M39.4.1 SIDE OPENING FRAME - NO ROBOT MOTION" if bool(fit.get("reliable", False)) else "REJECT: M39.4.1 OPENING RECONSTRUCTION UNCERTAIN",
        "operator_action": "validate_m39_4_1_opening_frame_before_robot_routing" if bool(fit.get("reliable", False)) else "inspect_m39_4_1_arc_and_opening_debug",
    })
    scene["selected_grasp_branch"] = str(summary["selected_grasp_branch"])
    scene["operator_action"] = str(summary["operator_action"])
    summary["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
    scene["m39_4_1_side_opening_reconstruction"] = summary
    return summary


def draw_m3941_side_opening_overlay(image_bgr: np.ndarray, summary: Mapping[str, Any]) -> np.ndarray:
    output = image_bgr.copy()
    fit = summary.get("selected") if isinstance(summary.get("selected"), Mapping) else summary.get("diagnostic")
    if not isinstance(fit, Mapping):
        return output
    opening_uv = fit.get("opening_center_uv")
    grasp_uv = fit.get("preview_grasp_center_uv")
    if not (isinstance(opening_uv, list) and len(opening_uv) == 2):
        return output
    po = np.asarray(opening_uv, dtype=np.float64)
    pg = np.asarray(grasp_uv, dtype=np.float64) if isinstance(grasp_uv, list) and len(grasp_uv) == 2 else po
    opening = fit.get("opening_center_camera_mm")
    frame = fit.get("side_grasp_frame_camera") if isinstance(fit.get("side_grasp_frame_camera"), Mapping) else {}
    if not (isinstance(opening, list) and len(opening) == 3):
        return output
    origin_3d = np.asarray(opening, dtype=np.float64)
    intrinsics = fit.get("_overlay_intrinsics") if isinstance(fit.get("_overlay_intrinsics"), Mapping) else None
    # Online caller normally draws without the private intrinsics helper; use
    # already projected grasp point and 2-D axis directions if available below.
    p_open = tuple(int(round(float(v))) for v in po)
    p_grasp = tuple(int(round(float(v))) for v in pg)
    reliable = bool(fit.get("reliable", False))
    cv2.circle(output, p_open, 9, (255, 0, 255) if reliable else (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(output, p_grasp, 7, (0, 255, 0) if reliable else (0, 128, 255), 2, cv2.LINE_AA)
    cv2.line(output, p_open, p_grasp, (255, 255, 0), 3, cv2.LINE_AA)
    x_tip = fit.get("frame_x_tip_uv")
    z_tip = fit.get("frame_z_tip_uv")
    if isinstance(x_tip, list) and len(x_tip) == 2:
        px = tuple(int(round(float(v))) for v in x_tip)
        cv2.arrowedLine(output, p_open, px, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(output, "+X close", (px[0] + 3, px[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    if isinstance(z_tip, list) and len(z_tip) == 2:
        pz = tuple(int(round(float(v))) for v in z_tip)
        cv2.arrowedLine(output, p_open, pz, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(output, "+Z insert", (pz[0] + 3, pz[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    # Use the source M39.4.0.1 axis projection for a stable 2-D insertion arrow.
    axis_angle = fit.get("axis_image_angle_deg_0_180")
    if axis_angle is None:
        # Recover from the opening->preview segment if projection has enough length.
        vector = pg - po
        if float(np.linalg.norm(vector)) > 1.0:
            axis_angle = float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0)
    text = "M39.4.1 OPENING FRAME" if reliable else "M39.4.1 OPENING UNCERTAIN"
    cv2.putText(output, text, (14, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 0, 255), 2, cv2.LINE_AA)
    arc = fit.get("camera_facing_outer_arc") if isinstance(fit.get("camera_facing_outer_arc"), Mapping) else {}
    support = fit.get("opening_support") if isinstance(fit.get("opening_support"), Mapping) else {}
    detail = f"arc={float(arc.get('arc_span_deg_p5_p95') or 0.0):.0f}deg drop={float(support.get('drop_ratio') or 0.0):.2f} robot=OFF"
    cv2.putText(output, detail, (14, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return output
