"""M37 parameterized 3-D template fitting for side-lying foam rings.

The target is modeled as a short hollow cylinder with known nominal outer
radius, inner radius and axial length.  Only the visible ``foam_ring`` RGB-D
points are required.  The implementation intentionally avoids a generic ICP
stack and SciPy/Open3D dependencies so it remains deployable in the existing
RK3576 Python environment.

The fitted axis is directed from the farther endpoint toward the endpoint that
is closer to the depth-camera origin.  A near-side upper-rim point is then
computed from the fitted 3-D template, rather than from the highest pixel of the
2-D segmentation contour.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .geometry import depth_pixels_to_points, project_point
from .segmentation import SegmentationInstance


_EPS = 1e-9


def _float(section: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(section.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _int(section: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(section.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= _EPS:
        raise ValueError("zero-length vector")
    return value / norm


def _basis_perpendicular(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(axis[2])) >= 0.90:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    first = _unit(np.cross(axis, reference))
    second = _unit(np.cross(axis, first))
    return first, second


def _fibonacci_hemisphere(count: int) -> Sequence[np.ndarray]:
    count = max(32, int(count))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions = []
    for index in range(count):
        # Axis sign is resolved after fitting, so only one hemisphere is needed.
        z = 1.0 - (float(index) + 0.5) / float(count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = float(index) * golden_angle
        directions.append(
            np.asarray(
                [radius * math.cos(angle), radius * math.sin(angle), z],
                dtype=np.float64,
            )
        )
    return directions


def _local_axis_candidates(
    axis: np.ndarray,
    maximum_angle_deg: float,
    radial_steps: int,
    azimuth_steps: int,
) -> Sequence[np.ndarray]:
    axis = _unit(axis)
    first, second = _basis_perpendicular(axis)
    candidates = [axis]
    radial_steps = max(1, int(radial_steps))
    azimuth_steps = max(4, int(azimuth_steps))
    for angle_deg in np.linspace(
        maximum_angle_deg / radial_steps,
        maximum_angle_deg,
        radial_steps,
    ):
        sine = math.sin(math.radians(float(angle_deg)))
        cosine = math.cos(math.radians(float(angle_deg)))
        for index in range(azimuth_steps):
            angle = 2.0 * math.pi * float(index) / float(azimuth_steps)
            candidate = cosine * axis + sine * (
                math.cos(angle) * first + math.sin(angle) * second
            )
            candidate = _unit(candidate)
            if candidate[2] < 0.0:
                candidate = -candidate
            candidates.append(candidate)
    return candidates


@dataclass(frozen=True)
class SideRingTemplateConfig:
    enabled: bool
    outer_radius_mm: float
    inner_radius_mm: float
    axial_length_mm: float
    mask_erode_px: int
    minimum_depth_mm: float
    maximum_depth_mm: float
    depth_lower_quantile: float
    depth_upper_quantile: float
    maximum_depth_behind_median_mm: float
    maximum_fit_points: int
    global_axis_samples: int
    local_refine_angles_deg: Tuple[float, ...]
    local_refine_radial_steps: int
    local_refine_azimuth_steps: int
    fixed_radius_iterations: int
    radial_inlier_threshold_mm: float
    minimum_radial_inlier_ratio: float
    maximum_radial_residual_median_mm: float
    maximum_radial_residual_p90_mm: float
    minimum_observed_axis_span_mm: float
    maximum_observed_axis_span_mm: float
    minimum_side_lay_angle_deg: float
    endpoint_quantile_low: float
    endpoint_quantile_high: float
    near_endpoint_metric: str
    top_arc_sample_count: int
    grasp_radius_mode: str
    grasp_axial_inset_mm: float
    random_seed: int

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "SideRingTemplateConfig":
        section = raw_config.get("side_ring_template") or {}
        if not isinstance(section, Mapping):
            raise ValueError("side_ring_template must be a mapping")
        object_geometry = raw_config.get("object_geometry") or {}
        if not isinstance(object_geometry, Mapping):
            object_geometry = {}
        nominal_outer = _float(object_geometry, "nominal_outer_diameter_mm", 85.0)
        nominal_inner = _float(object_geometry, "nominal_inner_diameter_mm", 60.0)
        axial_length = _float(object_geometry, "axial_length_mm", 70.0)
        refine_raw = section.get("local_refine_angles_deg", [12.0, 4.0, 1.5])
        if not isinstance(refine_raw, (list, tuple)):
            refine_raw = [12.0, 4.0, 1.5]
        refine_angles = tuple(max(0.1, float(item)) for item in refine_raw)
        return cls(
            enabled=bool(section.get("enabled", True)),
            outer_radius_mm=_float(section, "outer_radius_mm", nominal_outer / 2.0),
            inner_radius_mm=_float(section, "inner_radius_mm", nominal_inner / 2.0),
            axial_length_mm=_float(section, "axial_length_mm", axial_length),
            mask_erode_px=max(0, _int(section, "mask_erode_px", 2)),
            minimum_depth_mm=_float(section, "minimum_depth_mm", 150.0),
            maximum_depth_mm=_float(section, "maximum_depth_mm", 3000.0),
            depth_lower_quantile=min(0.25, max(0.0, _float(section, "depth_lower_quantile", 0.01))),
            depth_upper_quantile=min(1.0, max(0.75, _float(section, "depth_upper_quantile", 0.99))),
            maximum_depth_behind_median_mm=max(
                10.0,
                _float(section, "maximum_depth_behind_median_mm", 95.0),
            ),
            maximum_fit_points=max(200, _int(section, "maximum_fit_points", 1200)),
            global_axis_samples=max(32, _int(section, "global_axis_samples", 320)),
            local_refine_angles_deg=refine_angles,
            local_refine_radial_steps=max(1, _int(section, "local_refine_radial_steps", 3)),
            local_refine_azimuth_steps=max(4, _int(section, "local_refine_azimuth_steps", 16)),
            fixed_radius_iterations=max(3, _int(section, "fixed_radius_iterations", 12)),
            radial_inlier_threshold_mm=max(
                0.5,
                _float(section, "radial_inlier_threshold_mm", 6.0),
            ),
            minimum_radial_inlier_ratio=min(
                1.0,
                max(0.05, _float(section, "minimum_radial_inlier_ratio", 0.65)),
            ),
            maximum_radial_residual_median_mm=max(
                0.5,
                _float(section, "maximum_radial_residual_median_mm", 4.0),
            ),
            maximum_radial_residual_p90_mm=max(
                1.0,
                _float(section, "maximum_radial_residual_p90_mm", 16.0),
            ),
            minimum_observed_axis_span_mm=max(
                1.0,
                _float(section, "minimum_observed_axis_span_mm", 35.0),
            ),
            maximum_observed_axis_span_mm=max(
                20.0,
                _float(section, "maximum_observed_axis_span_mm", 105.0),
            ),
            minimum_side_lay_angle_deg=min(
                89.0,
                max(0.0, _float(section, "minimum_side_lay_angle_deg", 45.0)),
            ),
            endpoint_quantile_low=min(
                0.30,
                max(0.0, _float(section, "endpoint_quantile_low", 0.05)),
            ),
            endpoint_quantile_high=min(
                1.0,
                max(0.70, _float(section, "endpoint_quantile_high", 0.95)),
            ),
            near_endpoint_metric=str(
                section.get("near_endpoint_metric") or "euclidean_camera_distance"
            ),
            top_arc_sample_count=max(72, _int(section, "top_arc_sample_count", 720)),
            grasp_radius_mode=str(section.get("grasp_radius_mode") or "wall_midline"),
            grasp_axial_inset_mm=max(
                0.0,
                _float(section, "grasp_axial_inset_mm", 0.0),
            ),
            random_seed=_int(section, "random_seed", 3701),
        )


@dataclass
class _AxisEvaluation:
    score: float
    axis: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    circle_center_2d: np.ndarray
    axis_point: np.ndarray
    radial_distance_mm: np.ndarray
    radial_residual_mm: np.ndarray
    radial_inlier_mask: np.ndarray
    radial_inlier_ratio: float
    residual_median_mm: float
    residual_p70_mm: float
    residual_p90_mm: float
    axial_coordinate_mm: np.ndarray
    observed_axis_span_mm: float


def _fit_circle_center_fixed_radius(
    points_2d: np.ndarray,
    radius_mm: float,
    iterations: int,
) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    design = np.column_stack(
        (2.0 * points_2d[:, 0], 2.0 * points_2d[:, 1], np.ones(len(points_2d)))
    )
    target = np.sum(points_2d * points_2d, axis=1)
    try:
        center = np.linalg.lstsq(design, target, rcond=None)[0][:2]
    except np.linalg.LinAlgError:
        center = np.median(points_2d, axis=0)
    center = np.asarray(center, dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        difference = points_2d - center
        distance = np.maximum(np.linalg.norm(difference, axis=1), 1e-6)
        residual = distance - float(radius_mm)
        absolute = np.abs(residual)
        weights = np.ones_like(residual)
        huber_delta = 6.0
        outside = absolute > huber_delta
        weights[outside] = huber_delta / np.maximum(absolute[outside], 1e-6)
        weights[absolute > 20.0] *= 0.10
        jacobian = -difference / distance[:, None]
        hessian = jacobian.T @ (weights[:, None] * jacobian) + np.eye(2) * 1e-6
        gradient = jacobian.T @ (weights * residual)
        try:
            step = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            break
        step_norm = float(np.linalg.norm(step))
        if step_norm > 10.0:
            step *= 10.0 / step_norm
        center += step
        if float(np.linalg.norm(step)) < 1e-4:
            break
    return center


def _evaluate_axis(
    points: np.ndarray,
    axis: np.ndarray,
    config: SideRingTemplateConfig,
) -> _AxisEvaluation:
    axis = _unit(axis)
    basis_u, basis_v = _basis_perpendicular(axis)
    projected = np.column_stack((points @ basis_u, points @ basis_v))
    circle_center = _fit_circle_center_fixed_radius(
        projected,
        config.outer_radius_mm,
        config.fixed_radius_iterations,
    )
    radial_distance = np.linalg.norm(projected - circle_center, axis=1)
    radial_residual = np.abs(radial_distance - config.outer_radius_mm)
    radial_inlier_mask = radial_residual <= config.radial_inlier_threshold_mm
    radial_inlier_ratio = float(np.mean(radial_inlier_mask))
    residual_median = float(np.median(radial_residual))
    residual_p70 = float(np.percentile(radial_residual, 70))
    residual_p90 = float(np.percentile(radial_residual, 90))

    axial_coordinate = points @ axis
    axial_for_span = (
        axial_coordinate[radial_inlier_mask]
        if int(np.count_nonzero(radial_inlier_mask)) >= 20
        else axial_coordinate
    )
    observed_span = float(
        np.percentile(axial_for_span, 95) - np.percentile(axial_for_span, 5)
    )
    span_penalty = max(
        0.0,
        observed_span - config.maximum_observed_axis_span_mm,
    ) * 0.05
    span_penalty += max(
        0.0,
        config.minimum_observed_axis_span_mm - observed_span,
    ) * 0.03
    score = (
        residual_median
        + 0.35 * residual_p70
        + 0.06 * residual_p90
        + 8.0 * (1.0 - radial_inlier_ratio)
        + span_penalty
    )
    median_axial = float(np.median(axial_for_span))
    axis_point = (
        basis_u * float(circle_center[0])
        + basis_v * float(circle_center[1])
        + axis * median_axial
    )
    return _AxisEvaluation(
        score=float(score),
        axis=axis,
        basis_u=basis_u,
        basis_v=basis_v,
        circle_center_2d=circle_center,
        axis_point=axis_point,
        radial_distance_mm=radial_distance,
        radial_residual_mm=radial_residual,
        radial_inlier_mask=radial_inlier_mask,
        radial_inlier_ratio=radial_inlier_ratio,
        residual_median_mm=residual_median,
        residual_p70_mm=residual_p70,
        residual_p90_mm=residual_p90,
        axial_coordinate_mm=axial_coordinate,
        observed_axis_span_mm=observed_span,
    )


def _fit_axis(points: np.ndarray, config: SideRingTemplateConfig) -> _AxisEvaluation:
    rng = np.random.default_rng(config.random_seed)
    if len(points) > config.maximum_fit_points:
        indexes = rng.choice(len(points), size=config.maximum_fit_points, replace=False)
        search_points = points[indexes]
    else:
        search_points = points

    best: Optional[_AxisEvaluation] = None
    for axis in _fibonacci_hemisphere(config.global_axis_samples):
        candidate = _evaluate_axis(search_points, axis, config)
        if best is None or candidate.score < best.score:
            best = candidate
    assert best is not None

    for maximum_angle_deg in config.local_refine_angles_deg:
        refined: Optional[_AxisEvaluation] = None
        for axis in _local_axis_candidates(
            best.axis,
            maximum_angle_deg,
            config.local_refine_radial_steps,
            config.local_refine_azimuth_steps,
        ):
            candidate = _evaluate_axis(search_points, axis, config)
            if refined is None or candidate.score < refined.score:
                refined = candidate
        assert refined is not None
        best = refined

    return _evaluate_axis(points, best.axis, config)


def _trim_points_by_depth(
    points: np.ndarray,
    config: SideRingTemplateConfig,
) -> np.ndarray:
    if len(points) == 0:
        return points
    depth = points[:, 2]
    lower = float(np.quantile(depth, config.depth_lower_quantile))
    upper = float(np.quantile(depth, config.depth_upper_quantile))
    median = float(np.median(depth))
    upper = min(upper, median + config.maximum_depth_behind_median_mm)
    return points[(depth >= lower - 3.0) & (depth <= upper)]


def _project_points(
    points: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = points[:, 2] > 1e-6
    output[valid, 0] = (
        float(intrinsics["fx"]) * points[valid, 0] / points[valid, 2]
        + float(intrinsics["cx"])
    )
    output[valid, 1] = (
        float(intrinsics["fy"]) * points[valid, 1] / points[valid, 2]
        + float(intrinsics["cy"])
    )
    return output


def _camera_distance(point: np.ndarray, metric: str) -> float:
    if str(metric).strip().lower() in {"z", "depth", "camera_z"}:
        return float(point[2])
    return float(np.linalg.norm(point))


def _circle_points(
    center: np.ndarray,
    axis: np.ndarray,
    radius_mm: float,
    count: int,
) -> np.ndarray:
    first, second = _basis_perpendicular(axis)
    angles = np.linspace(0.0, 2.0 * math.pi, max(12, int(count)), endpoint=False)
    return center[None, :] + float(radius_mm) * (
        np.cos(angles)[:, None] * first[None, :]
        + np.sin(angles)[:, None] * second[None, :]
    )


def _top_arc_point(
    center: np.ndarray,
    axis: np.ndarray,
    radius_mm: float,
    intrinsics: Mapping[str, float],
    sample_count: int,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    points = _circle_points(center, axis, radius_mm, sample_count)
    pixels = _project_points(points, intrinsics)
    finite = np.isfinite(pixels).all(axis=1)
    if not np.any(finite):
        raise ValueError("near-side rim cannot be projected")
    valid_indexes = np.nonzero(finite)[0]
    selected = int(valid_indexes[np.argmin(pixels[finite, 1])])
    return points[selected], (float(pixels[selected, 0]), float(pixels[selected, 1]))


def fit_side_ring_instance(
    instance: SegmentationInstance,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    config: SideRingTemplateConfig,
    *,
    mouth_matched: bool = False,
) -> Dict[str, Any]:
    """Fit one parameterized short-cylinder template to a foam-ring mask."""

    started = time.perf_counter()
    if instance.class_name != "foam_ring":
        raise ValueError("fit_side_ring_instance requires foam_ring")
    mask = instance.mask.astype(np.uint8)
    if config.mask_erode_px > 0:
        kernel_size = config.mask_erode_px * 2 + 1
        mask = cv2.erode(
            mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
    points, pixels = depth_pixels_to_points(
        depth_mm,
        mask.astype(bool),
        intrinsics,
        config.minimum_depth_mm,
        config.maximum_depth_mm,
        stride=1,
    )
    raw_point_count = int(len(points))
    points = _trim_points_by_depth(points, config)
    trimmed_point_count = int(len(points))
    if trimmed_point_count < 80:
        return {
            "ring_instance_id": int(instance.instance_id),
            "mouth_matched": bool(mouth_matched),
            "eligible": False,
            "rejection_reasons": ["insufficient_depth_points"],
            "point_count_raw": raw_point_count,
            "point_count_trimmed": trimmed_point_count,
            "timing_ms": {"total_ms": (time.perf_counter() - started) * 1000.0},
        }

    fit_started = time.perf_counter()
    evaluation = _fit_axis(points, config)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0

    axial_inliers = evaluation.axial_coordinate_mm[evaluation.radial_inlier_mask]
    if len(axial_inliers) < 20:
        axial_inliers = evaluation.axial_coordinate_mm
    low = float(np.quantile(axial_inliers, config.endpoint_quantile_low))
    high = float(np.quantile(axial_inliers, config.endpoint_quantile_high))
    center_axial = 0.5 * (low + high)
    center = (
        evaluation.basis_u * float(evaluation.circle_center_2d[0])
        + evaluation.basis_v * float(evaluation.circle_center_2d[1])
        + evaluation.axis * center_axial
    )

    endpoint_positive = center + 0.5 * config.axial_length_mm * evaluation.axis
    endpoint_negative = center - 0.5 * config.axial_length_mm * evaluation.axis
    positive_distance = _camera_distance(endpoint_positive, config.near_endpoint_metric)
    negative_distance = _camera_distance(endpoint_negative, config.near_endpoint_metric)
    if positive_distance <= negative_distance:
        near_center = endpoint_positive
        far_center = endpoint_negative
    else:
        near_center = endpoint_negative
        far_center = endpoint_positive
    axis_toward_camera = _unit(near_center - far_center)

    center_view = _unit(center)
    axis_view_angle_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(float(np.dot(evaluation.axis, center_view))),
                    0.0,
                    1.0,
                )
            )
        )
    )

    if config.grasp_radius_mode.strip().lower() == "outer_surface":
        grasp_radius = config.outer_radius_mm
    elif config.grasp_radius_mode.strip().lower() == "inner_surface":
        grasp_radius = config.inner_radius_mm
    else:
        grasp_radius = 0.5 * (config.outer_radius_mm + config.inner_radius_mm)

    near_rim_top, near_rim_top_uv = _top_arc_point(
        near_center,
        axis_toward_camera,
        grasp_radius,
        intrinsics,
        config.top_arc_sample_count,
    )
    grasp_circle_center = near_center - axis_toward_camera * config.grasp_axial_inset_mm
    grasp_point, grasp_point_uv = _top_arc_point(
        grasp_circle_center,
        axis_toward_camera,
        grasp_radius,
        intrinsics,
        config.top_arc_sample_count,
    )

    fitted_radius = float(
        np.median(
            evaluation.radial_distance_mm[evaluation.radial_inlier_mask]
            if np.any(evaluation.radial_inlier_mask)
            else evaluation.radial_distance_mm
        )
    )
    rejection_reasons = []
    if mouth_matched:
        rejection_reasons.append("mouth_matched_prefer_m36_branch")
    if evaluation.radial_inlier_ratio < config.minimum_radial_inlier_ratio:
        rejection_reasons.append("radial_inlier_ratio_too_low")
    if evaluation.residual_median_mm > config.maximum_radial_residual_median_mm:
        rejection_reasons.append("radial_residual_median_too_high")
    if evaluation.residual_p90_mm > config.maximum_radial_residual_p90_mm:
        rejection_reasons.append("radial_residual_p90_too_high")
    if evaluation.observed_axis_span_mm < config.minimum_observed_axis_span_mm:
        rejection_reasons.append("observed_axis_span_too_short")
    if evaluation.observed_axis_span_mm > config.maximum_observed_axis_span_mm:
        rejection_reasons.append("observed_axis_span_too_long")
    if axis_view_angle_deg < config.minimum_side_lay_angle_deg:
        rejection_reasons.append("axis_not_side_laying")

    center_uv = project_point(center, intrinsics)
    near_center_uv = project_point(near_center, intrinsics)
    far_center_uv = project_point(far_center, intrinsics)
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": "1.0",
        "message_type": "side_ring_parameterized_template_fit",
        "ring_instance_id": int(instance.instance_id),
        "ring_confidence": float(instance.confidence),
        "ring_bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "mouth_matched": bool(mouth_matched),
        "eligible": len(rejection_reasons) == 0,
        "rejection_reasons": rejection_reasons,
        "fit_score": float(evaluation.score),
        "point_count_raw": raw_point_count,
        "point_count_trimmed": trimmed_point_count,
        "radial_inlier_count": int(np.count_nonzero(evaluation.radial_inlier_mask)),
        "radial_inlier_ratio": float(evaluation.radial_inlier_ratio),
        "radial_residual_median_mm": float(evaluation.residual_median_mm),
        "radial_residual_p70_mm": float(evaluation.residual_p70_mm),
        "radial_residual_p90_mm": float(evaluation.residual_p90_mm),
        "outer_radius_nominal_mm": float(config.outer_radius_mm),
        "outer_radius_fitted_mm": fitted_radius,
        "inner_radius_nominal_mm": float(config.inner_radius_mm),
        "axial_length_nominal_mm": float(config.axial_length_mm),
        "observed_axis_span_mm": float(evaluation.observed_axis_span_mm),
        "axis_view_angle_deg": float(axis_view_angle_deg),
        "axis_direction_rule": "far_endpoint_to_camera_nearest_endpoint",
        "axis_toward_camera": axis_toward_camera.tolist(),
        "center_camera_mm": center.tolist(),
        "near_opening_center_camera_mm": near_center.tolist(),
        "far_opening_center_camera_mm": far_center.tolist(),
        "center_uv": list(center_uv) if center_uv is not None else None,
        "near_opening_center_uv": list(near_center_uv) if near_center_uv is not None else None,
        "far_opening_center_uv": list(far_center_uv) if far_center_uv is not None else None,
        "near_endpoint_camera_distance_mm": float(
            _camera_distance(near_center, config.near_endpoint_metric)
        ),
        "far_endpoint_camera_distance_mm": float(
            _camera_distance(far_center, config.near_endpoint_metric)
        ),
        "top_arc": {
            "definition": "near_opening_rim_wall_midline_highest_projected_point",
            "radius_mm": float(grasp_radius),
            "near_rim_top_camera_mm": near_rim_top.tolist(),
            "near_rim_top_uv": [float(near_rim_top_uv[0]), float(near_rim_top_uv[1])],
            "grasp_axial_inset_mm": float(config.grasp_axial_inset_mm),
            "grasp_point_camera_mm": grasp_point.tolist(),
            "grasp_point_uv": [float(grasp_point_uv[0]), float(grasp_point_uv[1])],
        },
        "timing_ms": {
            "axis_template_fit_ms": float(fit_ms),
            "total_ms": float(total_ms),
        },
        "_debug": {
            "trimmed_points_camera_mm": points,
            "radial_inlier_mask": evaluation.radial_inlier_mask,
            "near_outer_circle_camera_mm": _circle_points(
                near_center,
                axis_toward_camera,
                config.outer_radius_mm,
                96,
            ),
            "near_inner_circle_camera_mm": _circle_points(
                near_center,
                axis_toward_camera,
                config.inner_radius_mm,
                96,
            ),
            "far_outer_circle_camera_mm": _circle_points(
                far_center,
                axis_toward_camera,
                config.outer_radius_mm,
                96,
            ),
        },
    }


def select_best_side_ring(fits: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    eligible = [item for item in fits if bool(item.get("eligible", False))]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            float(item.get("near_endpoint_camera_distance_mm", float("inf"))),
            float(item.get("fit_score", float("inf"))),
            -float(item.get("ring_confidence", 0.0)),
        ),
    )
