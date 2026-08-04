"""M37 parameterized 3-D template fitting for side-lying foam rings.

The target is modeled as a short hollow cylinder with known nominal outer
radius, inner radius and axial length.  Only the visible ``foam_ring`` RGB-D
points are required.  The implementation intentionally avoids a generic ICP
stack and SciPy/Open3D dependencies so it remains deployable in the existing
RK3576 Python environment.

The fitted axis is directed from the farther endpoint toward the endpoint that
is closer to the depth-camera origin.  M37.1 computes the grasp point on the
camera-visible cylindrical crown, slightly inset from the near opening.  This
is intentionally different from the old M37 diagnostic point on the highest
projected near-opening rim pixel.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .geometry import project_point
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


@lru_cache(maxsize=16)
def _fibonacci_hemisphere(count: int) -> Tuple[np.ndarray, ...]:
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
    return tuple(directions)


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
    execution_mode: str
    stop_after_first_eligible: bool
    maximum_instances_to_attempt: int
    first_valid_search_profile: str
    exhaustive_search_profile: str
    point_extraction_bbox_padding_px: int
    minimum_depth_points: int
    fast_search_enabled: bool
    fast_maximum_fit_points: int
    fast_global_maximum_fit_points: int
    fast_global_axis_samples: int
    fast_local_refine_angles_deg: Tuple[float, ...]
    fast_local_refine_radial_steps: int
    fast_local_refine_azimuth_steps: int
    fast_fixed_radius_iterations: int
    fast_accept_max_score: float
    local_accurate_maximum_fit_points: int
    local_accurate_refine_angles_deg: Tuple[float, ...]
    local_accurate_refine_radial_steps: int
    local_accurate_refine_azimuth_steps: int
    local_accurate_fixed_radius_iterations: int
    accurate_fallback_enabled: bool
    accurate_fallback_on_quality_gate_failure: bool
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
    visible_crown_enabled: bool
    visible_crown_axial_band_start_mm: float
    visible_crown_axial_band_end_mm: float
    visible_crown_minimum_points: int
    visible_crown_angular_trim_quantile: float
    visible_crown_upper_fraction: float
    visible_crown_minimum_span_deg: float
    visible_crown_maximum_span_deg: float
    visible_crown_fallback_mode: str
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
        fast_refine_raw = section.get("fast_local_refine_angles_deg", [12.0, 4.0, 1.5])
        if not isinstance(fast_refine_raw, (list, tuple)):
            fast_refine_raw = [12.0, 4.0, 1.5]
        fast_refine_angles = tuple(max(0.1, float(item)) for item in fast_refine_raw)
        local_accurate_raw = section.get(
            "local_accurate_refine_angles_deg", [4.0, 1.5, 0.5]
        )
        if not isinstance(local_accurate_raw, (list, tuple)):
            local_accurate_raw = [4.0, 1.5, 0.5]
        local_accurate_angles = tuple(
            max(0.1, float(item)) for item in local_accurate_raw
        )
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
            execution_mode=str(
                section.get("execution_mode") or "first_valid_confidence"
            ).strip().lower(),
            stop_after_first_eligible=bool(
                section.get("stop_after_first_eligible", True)
            ),
            maximum_instances_to_attempt=max(
                0, _int(section, "maximum_instances_to_attempt", 0)
            ),
            first_valid_search_profile=str(
                section.get("first_valid_search_profile") or "auto"
            ).strip().lower(),
            exhaustive_search_profile=str(
                section.get("exhaustive_search_profile") or "accurate"
            ).strip().lower(),
            point_extraction_bbox_padding_px=max(
                0, _int(section, "point_extraction_bbox_padding_px", 3)
            ),
            minimum_depth_points=max(
                20, _int(section, "minimum_depth_points", 80)
            ),
            fast_search_enabled=bool(section.get("fast_search_enabled", True)),
            fast_maximum_fit_points=max(
                200, _int(section, "fast_maximum_fit_points", 700)
            ),
            fast_global_maximum_fit_points=max(
                120, _int(section, "fast_global_maximum_fit_points", 400)
            ),
            fast_global_axis_samples=max(
                32, _int(section, "fast_global_axis_samples", 80)
            ),
            fast_local_refine_angles_deg=fast_refine_angles,
            fast_local_refine_radial_steps=max(
                1, _int(section, "fast_local_refine_radial_steps", 2)
            ),
            fast_local_refine_azimuth_steps=max(
                4, _int(section, "fast_local_refine_azimuth_steps", 12)
            ),
            fast_fixed_radius_iterations=max(
                3, _int(section, "fast_fixed_radius_iterations", 6)
            ),
            fast_accept_max_score=max(
                0.5, _float(section, "fast_accept_max_score", 3.0)
            ),
            local_accurate_maximum_fit_points=max(
                200, _int(section, "local_accurate_maximum_fit_points", 1200)
            ),
            local_accurate_refine_angles_deg=local_accurate_angles,
            local_accurate_refine_radial_steps=max(
                1, _int(section, "local_accurate_refine_radial_steps", 3)
            ),
            local_accurate_refine_azimuth_steps=max(
                4, _int(section, "local_accurate_refine_azimuth_steps", 16)
            ),
            local_accurate_fixed_radius_iterations=max(
                3, _int(section, "local_accurate_fixed_radius_iterations", 12)
            ),
            accurate_fallback_enabled=bool(
                section.get("accurate_fallback_enabled", True)
            ),
            accurate_fallback_on_quality_gate_failure=bool(
                section.get("accurate_fallback_on_quality_gate_failure", True)
            ),
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
            grasp_radius_mode=str(section.get("grasp_radius_mode") or "outer_surface"),
            grasp_axial_inset_mm=max(
                0.0,
                _float(section, "grasp_axial_inset_mm", 11.0),
            ),
            visible_crown_enabled=bool(section.get("visible_crown_enabled", True)),
            visible_crown_axial_band_start_mm=max(
                0.0,
                _float(section, "visible_crown_axial_band_start_mm", 5.0),
            ),
            visible_crown_axial_band_end_mm=max(
                1.0,
                _float(section, "visible_crown_axial_band_end_mm", 22.0),
            ),
            visible_crown_minimum_points=max(
                20,
                _int(section, "visible_crown_minimum_points", 80),
            ),
            visible_crown_angular_trim_quantile=min(
                0.20,
                max(
                    0.0,
                    _float(section, "visible_crown_angular_trim_quantile", 0.01),
                ),
            ),
            visible_crown_upper_fraction=min(
                1.0,
                max(0.0, _float(section, "visible_crown_upper_fraction", 0.44)),
            ),
            visible_crown_minimum_span_deg=max(
                5.0,
                _float(section, "visible_crown_minimum_span_deg", 35.0),
            ),
            visible_crown_maximum_span_deg=min(
                355.0,
                max(
                    30.0,
                    _float(section, "visible_crown_maximum_span_deg", 175.0),
                ),
            ),
            visible_crown_fallback_mode=str(
                section.get("visible_crown_fallback_mode") or "camera_facing"
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
    *,
    fixed_radius_iterations: Optional[int] = None,
) -> _AxisEvaluation:
    axis = _unit(axis)
    basis_u, basis_v = _basis_perpendicular(axis)
    projected = np.column_stack((points @ basis_u, points @ basis_v))
    circle_center = _fit_circle_center_fixed_radius(
        projected,
        config.outer_radius_mm,
        (
            config.fixed_radius_iterations
            if fixed_radius_iterations is None
            else int(fixed_radius_iterations)
        ),
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


def _sample_search_points(
    points: np.ndarray,
    maximum_points: int,
    *,
    random_seed: int,
) -> np.ndarray:
    if len(points) <= maximum_points:
        return points
    rng = np.random.default_rng(random_seed)
    indexes = rng.choice(len(points), size=maximum_points, replace=False)
    return points[indexes]


def _fit_axis_profile(
    points: np.ndarray,
    config: SideRingTemplateConfig,
    *,
    profile: str,
) -> Tuple[_AxisEvaluation, Dict[str, Any]]:
    profile = str(profile).strip().lower()
    fast = profile == "fast"
    local_maximum_fit_points = (
        config.fast_maximum_fit_points if fast else config.maximum_fit_points
    )
    global_maximum_fit_points = (
        min(config.fast_global_maximum_fit_points, local_maximum_fit_points)
        if fast else local_maximum_fit_points
    )
    global_axis_samples = (
        config.fast_global_axis_samples if fast else config.global_axis_samples
    )
    local_refine_angles = (
        config.fast_local_refine_angles_deg
        if fast
        else config.local_refine_angles_deg
    )
    local_radial_steps = (
        config.fast_local_refine_radial_steps
        if fast
        else config.local_refine_radial_steps
    )
    local_azimuth_steps = (
        config.fast_local_refine_azimuth_steps
        if fast
        else config.local_refine_azimuth_steps
    )
    fixed_iterations = (
        config.fast_fixed_radius_iterations
        if fast
        else config.fixed_radius_iterations
    )

    timing: Dict[str, Any] = {
        "profile": profile,
        "maximum_fit_points": int(local_maximum_fit_points),
        "global_maximum_fit_points": int(global_maximum_fit_points),
        "global_axis_samples": int(global_axis_samples),
        "fixed_radius_iterations": int(fixed_iterations),
        "candidate_evaluations": 0,
        "local_refine_levels_ms": [],
    }
    profile_started = time.perf_counter()
    sampling_started = time.perf_counter()
    local_points = _sample_search_points(
        points,
        local_maximum_fit_points,
        random_seed=config.random_seed,
    )
    global_points = _sample_search_points(
        local_points,
        global_maximum_fit_points,
        random_seed=config.random_seed + 17,
    )
    timing["sampling_ms"] = (time.perf_counter() - sampling_started) * 1000.0
    timing["global_search_point_count"] = int(len(global_points))
    timing["local_search_point_count"] = int(len(local_points))
    # Historical compatibility field.
    timing["search_point_count"] = int(len(local_points))

    global_started = time.perf_counter()
    best: Optional[_AxisEvaluation] = None
    for axis in _fibonacci_hemisphere(global_axis_samples):
        candidate = _evaluate_axis(
            global_points,
            axis,
            config,
            fixed_radius_iterations=fixed_iterations,
        )
        timing["candidate_evaluations"] += 1
        if best is None or candidate.score < best.score:
            best = candidate
    assert best is not None
    timing["global_search_ms"] = (time.perf_counter() - global_started) * 1000.0

    # Re-evaluate the coarse winner on the larger local-search sample before
    # angular refinement.  This makes the 400-point global stage cheap without
    # allowing a sparse-sample winner to dominate the final result.
    best = _evaluate_axis(
        local_points,
        best.axis,
        config,
        fixed_radius_iterations=fixed_iterations,
    )
    timing["candidate_evaluations"] += 1

    local_total = 0.0
    for maximum_angle_deg in local_refine_angles:
        level_started = time.perf_counter()
        refined: Optional[_AxisEvaluation] = None
        candidates = _local_axis_candidates(
            best.axis,
            maximum_angle_deg,
            local_radial_steps,
            local_azimuth_steps,
        )
        for axis in candidates:
            candidate = _evaluate_axis(
                local_points,
                axis,
                config,
                fixed_radius_iterations=fixed_iterations,
            )
            timing["candidate_evaluations"] += 1
            if refined is None or candidate.score < refined.score:
                refined = candidate
        assert refined is not None
        best = refined
        level_ms = (time.perf_counter() - level_started) * 1000.0
        local_total += level_ms
        timing["local_refine_levels_ms"].append(
            {
                "maximum_angle_deg": float(maximum_angle_deg),
                "candidate_count": int(len(candidates)),
                "elapsed_ms": float(level_ms),
            }
        )
    timing["local_refine_ms"] = float(local_total)

    final_started = time.perf_counter()
    final = _evaluate_axis(
        points,
        best.axis,
        config,
        fixed_radius_iterations=(
            config.fixed_radius_iterations if fast else fixed_iterations
        ),
    )
    timing["final_full_point_evaluation_ms"] = (
        time.perf_counter() - final_started
    ) * 1000.0
    timing["total_ms"] = (time.perf_counter() - profile_started) * 1000.0
    return final, timing


def _fit_axis_local_accurate(
    points: np.ndarray,
    config: SideRingTemplateConfig,
    *,
    initial_axis: np.ndarray,
) -> Tuple[_AxisEvaluation, Dict[str, Any]]:
    """Accurate warm-start refinement without repeating global direction search."""

    timing: Dict[str, Any] = {
        "profile": "local_accurate",
        "warm_start": True,
        "global_axis_samples": 0,
        "global_search_ms": 0.0,
        "candidate_evaluations": 0,
        "local_refine_levels_ms": [],
        "maximum_fit_points": int(config.local_accurate_maximum_fit_points),
        "fixed_radius_iterations": int(config.local_accurate_fixed_radius_iterations),
    }
    started = time.perf_counter()
    sample_started = time.perf_counter()
    search_points = _sample_search_points(
        points,
        config.local_accurate_maximum_fit_points,
        random_seed=config.random_seed + 31,
    )
    timing["sampling_ms"] = (time.perf_counter() - sample_started) * 1000.0
    timing["search_point_count"] = int(len(search_points))

    best = _evaluate_axis(
        search_points,
        _unit(np.asarray(initial_axis, dtype=np.float64)),
        config,
        fixed_radius_iterations=config.local_accurate_fixed_radius_iterations,
    )
    timing["candidate_evaluations"] += 1

    local_total = 0.0
    for maximum_angle_deg in config.local_accurate_refine_angles_deg:
        level_started = time.perf_counter()
        candidates = _local_axis_candidates(
            best.axis,
            maximum_angle_deg,
            config.local_accurate_refine_radial_steps,
            config.local_accurate_refine_azimuth_steps,
        )
        refined: Optional[_AxisEvaluation] = None
        for axis in candidates:
            candidate = _evaluate_axis(
                search_points,
                axis,
                config,
                fixed_radius_iterations=config.local_accurate_fixed_radius_iterations,
            )
            timing["candidate_evaluations"] += 1
            if refined is None or candidate.score < refined.score:
                refined = candidate
        assert refined is not None
        best = refined
        elapsed = (time.perf_counter() - level_started) * 1000.0
        local_total += elapsed
        timing["local_refine_levels_ms"].append(
            {
                "maximum_angle_deg": float(maximum_angle_deg),
                "candidate_count": int(len(candidates)),
                "elapsed_ms": float(elapsed),
            }
        )
    timing["local_refine_ms"] = float(local_total)

    final_started = time.perf_counter()
    final = _evaluate_axis(
        points,
        best.axis,
        config,
        fixed_radius_iterations=config.local_accurate_fixed_radius_iterations,
    )
    timing["final_full_point_evaluation_ms"] = (
        time.perf_counter() - final_started
    ) * 1000.0
    timing["total_ms"] = (time.perf_counter() - started) * 1000.0
    return final, timing

def _fast_fit_fallback_reasons(
    evaluation: _AxisEvaluation,
    config: SideRingTemplateConfig,
) -> List[str]:
    reasons: List[str] = []
    if evaluation.score > config.fast_accept_max_score:
        reasons.append("fast_fit_score_above_accept_threshold")
    if config.accurate_fallback_on_quality_gate_failure:
        if evaluation.radial_inlier_ratio < config.minimum_radial_inlier_ratio:
            reasons.append("fast_radial_inlier_ratio_gate_failed")
        if evaluation.residual_median_mm > config.maximum_radial_residual_median_mm:
            reasons.append("fast_radial_residual_median_gate_failed")
        if evaluation.residual_p90_mm > config.maximum_radial_residual_p90_mm:
            reasons.append("fast_radial_residual_p90_gate_failed")
        if evaluation.observed_axis_span_mm < config.minimum_observed_axis_span_mm:
            reasons.append("fast_axis_span_short_gate_failed")
        if evaluation.observed_axis_span_mm > config.maximum_observed_axis_span_mm:
            reasons.append("fast_axis_span_long_gate_failed")
    return reasons


def _fit_axis(
    points: np.ndarray,
    config: SideRingTemplateConfig,
    *,
    search_profile: str = "auto",
    initial_axis: Optional[np.ndarray] = None,
) -> Tuple[_AxisEvaluation, Dict[str, Any]]:
    requested = str(search_profile or "auto").strip().lower()
    if requested not in {"auto", "fast", "accurate", "local_accurate"}:
        raise ValueError(
            "search_profile must be auto, fast, accurate or local_accurate"
        )
    if requested == "local_accurate":
        if initial_axis is None:
            raise ValueError("local_accurate search requires initial_axis")
        evaluation, local_timing = _fit_axis_local_accurate(
            points,
            config,
            initial_axis=np.asarray(initial_axis, dtype=np.float64),
        )
        return evaluation, {
            "requested_profile": requested,
            "final_profile": "local_accurate",
            "fallback_used": False,
            "fallback_reasons": [],
            "fast": None,
            "accurate": None,
            "local_accurate": local_timing,
            "total_ms": float(local_timing["total_ms"]),
        }

    use_fast = requested in {"auto", "fast"} and config.fast_search_enabled
    if not use_fast:
        evaluation, accurate_timing = _fit_axis_profile(
            points, config, profile="accurate"
        )
        return evaluation, {
            "requested_profile": requested,
            "final_profile": "accurate",
            "fallback_used": False,
            "fallback_reasons": [],
            "fast": None,
            "accurate": accurate_timing,
            "local_accurate": None,
            "total_ms": float(accurate_timing["total_ms"]),
        }

    started = time.perf_counter()
    fast_evaluation, fast_timing = _fit_axis_profile(
        points, config, profile="fast"
    )
    fallback_reasons = _fast_fit_fallback_reasons(fast_evaluation, config)
    fallback_used = bool(
        requested == "auto"
        and config.accurate_fallback_enabled
        and fallback_reasons
    )
    accurate_timing = None
    final_evaluation = fast_evaluation
    final_profile = "fast"
    if fallback_used:
        # Historical standalone/offline auto mode retains the full accurate
        # fallback.  The M37.4 online hybrid path explicitly requests ``fast``
        # for every candidate and performs at most one warm-start
        # ``local_accurate`` refinement at the scene level.
        final_evaluation, accurate_timing = _fit_axis_profile(
            points, config, profile="accurate"
        )
        final_profile = "accurate_fallback"
    return final_evaluation, {
        "requested_profile": requested,
        "final_profile": final_profile,
        "fallback_used": bool(fallback_used),
        "fallback_reasons": fallback_reasons,
        "fast": fast_timing,
        "accurate": accurate_timing,
        "local_accurate": None,
        "total_ms": (time.perf_counter() - started) * 1000.0,
    }

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


def _camera_facing_radial_direction(
    center: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Return the radial direction on the cylinder that faces the camera.

    The camera origin is ``[0, 0, 0]`` in the color optical frame.  Removing
    the component parallel to the cylinder axis leaves a vector in the local
    cross-section plane.
    """

    axis = _unit(axis)
    to_camera = -np.asarray(center, dtype=np.float64).reshape(3)
    radial = to_camera - axis * float(np.dot(to_camera, axis))
    return _unit(radial)


def _visible_crown_direction(
    points: np.ndarray,
    radial_inlier_mask: np.ndarray,
    near_center: np.ndarray,
    axis_toward_camera: np.ndarray,
    grasp_circle_center: np.ndarray,
    grasp_radius_mm: float,
    intrinsics: Mapping[str, float],
    config: SideRingTemplateConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate the camera-visible upper cylindrical crown direction.

    The visible cylinder points form a contiguous angular interval in a plane
    perpendicular to the fitted axis.  We recover that interval using the
    largest circular gap, robustly trim its two endpoints, identify the endpoint
    that projects higher in the image, then move a configurable fraction toward
    the lower endpoint.  The fraction is deliberately configurable because it
    describes the desired physical contact location on the visible arc, not a
    mathematical property of the cylinder.
    """

    axis = _unit(axis_toward_camera)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    radial_inlier_mask = np.asarray(radial_inlier_mask, dtype=bool).reshape(-1)
    if len(points) != len(radial_inlier_mask):
        raise ValueError("radial inlier mask length does not match point count")

    fallback_direction = _camera_facing_radial_direction(grasp_circle_center, axis)
    fallback_debug: Dict[str, Any] = {
        "direction_source": "camera_facing_fallback",
        "visible_point_count": 0,
        "visible_angular_span_deg": None,
        "upper_fraction": float(config.visible_crown_upper_fraction),
        "angular_trim_quantile": float(config.visible_crown_angular_trim_quantile),
    }
    if not config.visible_crown_enabled:
        fallback_debug["fallback_reason"] = "visible_crown_disabled"
        return fallback_direction, fallback_debug

    start_mm = float(config.visible_crown_axial_band_start_mm)
    end_mm = max(start_mm + 1.0, float(config.visible_crown_axial_band_end_mm))
    axial_inset_mm = np.dot(near_center[None, :] - points, axis)
    selected_mask = (
        radial_inlier_mask
        & (axial_inset_mm >= start_mm)
        & (axial_inset_mm <= end_mm)
    )
    selected_points = points[selected_mask]
    selected_axial = axial_inset_mm[selected_mask]
    fallback_debug["visible_point_count"] = int(len(selected_points))
    fallback_debug["axial_band_start_mm"] = start_mm
    fallback_debug["axial_band_end_mm"] = end_mm
    if len(selected_points) < config.visible_crown_minimum_points:
        fallback_debug["fallback_reason"] = "insufficient_visible_crown_points"
        return fallback_direction, fallback_debug

    axis_points = near_center[None, :] - selected_axial[:, None] * axis[None, :]
    radial_vectors = selected_points - axis_points
    radial_norms = np.linalg.norm(radial_vectors, axis=1)
    valid = radial_norms > _EPS
    radial_vectors = radial_vectors[valid] / radial_norms[valid, None]
    if len(radial_vectors) < config.visible_crown_minimum_points:
        fallback_debug["fallback_reason"] = "insufficient_nonzero_radial_vectors"
        return fallback_direction, fallback_debug

    basis_u, basis_v = _basis_perpendicular(axis)
    angles = np.mod(
        np.arctan2(radial_vectors @ basis_v, radial_vectors @ basis_u),
        2.0 * math.pi,
    )
    sorted_angles = np.sort(angles)
    circular_gaps = np.diff(
        np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * math.pi))
    )
    largest_gap_index = int(np.argmax(circular_gaps))
    occupied_start = float(sorted_angles[(largest_gap_index + 1) % len(sorted_angles)])
    unwrapped = np.mod(angles - occupied_start, 2.0 * math.pi)

    trim = float(config.visible_crown_angular_trim_quantile)
    lower = float(np.quantile(unwrapped, trim))
    upper = float(np.quantile(unwrapped, 1.0 - trim))
    angular_span = max(0.0, upper - lower)
    angular_span_deg = math.degrees(angular_span)
    fallback_debug["visible_angular_span_deg"] = float(angular_span_deg)
    if angular_span_deg < config.visible_crown_minimum_span_deg:
        fallback_debug["fallback_reason"] = "visible_angular_span_too_small"
        return fallback_direction, fallback_debug
    if angular_span_deg > config.visible_crown_maximum_span_deg:
        fallback_debug["fallback_reason"] = "visible_angular_span_too_large"
        return fallback_direction, fallback_debug

    angle_a = occupied_start + lower
    angle_b = occupied_start + upper
    direction_a = _unit(math.cos(angle_a) * basis_u + math.sin(angle_a) * basis_v)
    direction_b = _unit(math.cos(angle_b) * basis_u + math.sin(angle_b) * basis_v)
    endpoint_points = np.asarray(
        [
            grasp_circle_center + float(grasp_radius_mm) * direction_a,
            grasp_circle_center + float(grasp_radius_mm) * direction_b,
        ],
        dtype=np.float64,
    )
    endpoint_pixels = _project_points(endpoint_points, intrinsics)
    finite = np.isfinite(endpoint_pixels).all(axis=1)
    if not bool(np.all(finite)):
        fallback_debug["fallback_reason"] = "visible_arc_endpoints_not_projectable"
        return fallback_direction, fallback_debug

    # Parameterize from the endpoint that is visually higher toward the lower
    # endpoint.  This makes ``upper_fraction`` stable when the fitted basis or
    # axis sign changes while preserving the requested near-end axis direction.
    upper_fraction = float(config.visible_crown_upper_fraction)
    if float(endpoint_pixels[0, 1]) <= float(endpoint_pixels[1, 1]):
        selected_angle = angle_a + angular_span * upper_fraction
        upper_endpoint_angle = angle_a
        lower_endpoint_angle = angle_b
        upper_endpoint_uv = endpoint_pixels[0]
        lower_endpoint_uv = endpoint_pixels[1]
    else:
        selected_angle = angle_b - angular_span * upper_fraction
        upper_endpoint_angle = angle_b
        lower_endpoint_angle = angle_a
        upper_endpoint_uv = endpoint_pixels[1]
        lower_endpoint_uv = endpoint_pixels[0]

    direction = _unit(
        math.cos(selected_angle) * basis_u + math.sin(selected_angle) * basis_v
    )
    camera_facing = _camera_facing_radial_direction(grasp_circle_center, axis)
    if float(np.dot(direction, camera_facing)) <= 0.0:
        fallback_debug["fallback_reason"] = "selected_visible_arc_faces_away_from_camera"
        return fallback_direction, fallback_debug

    debug = {
        "direction_source": "visible_surface_angular_interval",
        "visible_point_count": int(len(radial_vectors)),
        "axial_band_start_mm": start_mm,
        "axial_band_end_mm": end_mm,
        "visible_angular_span_deg": float(angular_span_deg),
        "angular_trim_quantile": trim,
        "upper_fraction": upper_fraction,
        "upper_endpoint_angle_deg": float(math.degrees(upper_endpoint_angle) % 360.0),
        "lower_endpoint_angle_deg": float(math.degrees(lower_endpoint_angle) % 360.0),
        "selected_angle_deg": float(math.degrees(selected_angle) % 360.0),
        "upper_endpoint_uv": [
            float(upper_endpoint_uv[0]),
            float(upper_endpoint_uv[1]),
        ],
        "lower_endpoint_uv": [
            float(lower_endpoint_uv[0]),
            float(lower_endpoint_uv[1]),
        ],
        "camera_facing_dot": float(np.dot(direction, camera_facing)),
    }
    return direction, debug



def _extract_instance_points(
    instance: SegmentationInstance,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    config: SideRingTemplateConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    height, width = depth_mm.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in instance.bbox_xyxy)
    padding = max(config.point_extraction_bbox_padding_px, config.mask_erode_px + 1)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            {"bbox_xyxy": [x1, y1, x2, y2], "total_ms": 0.0},
        )

    mask_started = time.perf_counter()
    local_mask = instance.mask[y1:y2, x1:x2].astype(np.uint8, copy=True)
    if config.mask_erode_px > 0:
        kernel_size = config.mask_erode_px * 2 + 1
        local_mask = cv2.erode(
            local_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
    mask_ms = (time.perf_counter() - mask_started) * 1000.0

    deproject_started = time.perf_counter()
    local_depth = depth_mm[y1:y2, x1:x2]
    valid = (
        local_mask.astype(bool)
        & (local_depth >= config.minimum_depth_mm)
        & (local_depth <= config.maximum_depth_mm)
    )
    local_y, local_x = np.nonzero(valid)
    if local_x.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
        pixels = np.empty((0, 2), dtype=np.int32)
    else:
        xs = local_x + x1
        ys = local_y + y1
        z = local_depth[local_y, local_x].astype(np.float64)
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        points = np.column_stack((x, y, z))
        pixels = np.column_stack((xs, ys)).astype(np.int32)
    deproject_ms = (time.perf_counter() - deproject_started) * 1000.0
    return points, pixels, {
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "mask_prepare_ms": float(mask_ms),
        "depth_deproject_ms": float(deproject_ms),
        "total_ms": (time.perf_counter() - started) * 1000.0,
    }

def fit_side_ring_instance(
    instance: SegmentationInstance,
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    config: SideRingTemplateConfig,
    *,
    mouth_matched: bool = False,
    search_profile: str = "auto",
    initial_axis: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit one parameterized short-cylinder template to a foam-ring mask."""

    started = time.perf_counter()
    if instance.class_name != "foam_ring":
        raise ValueError("fit_side_ring_instance requires foam_ring")
    extract_started = time.perf_counter()
    points, pixels, extraction_timing = _extract_instance_points(
        instance, depth_mm, intrinsics, config
    )
    extraction_ms = (time.perf_counter() - extract_started) * 1000.0
    raw_point_count = int(len(points))
    trim_started = time.perf_counter()
    points = _trim_points_by_depth(points, config)
    depth_trim_ms = (time.perf_counter() - trim_started) * 1000.0
    trimmed_point_count = int(len(points))
    if trimmed_point_count < config.minimum_depth_points:
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ring_instance_id": int(instance.instance_id),
            "ring_confidence": float(instance.confidence),
            "ring_bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
            "mouth_matched": bool(mouth_matched),
            "eligible": False,
            "rejection_reasons": ["insufficient_depth_points"],
            "point_count_raw": raw_point_count,
            "point_count_trimmed": trimmed_point_count,
            "search_profile_requested": str(search_profile),
            "search_profile_used": None,
            "timing_ms": {
                "point_extraction_ms": float(extraction_ms),
                "mask_prepare_ms": float(extraction_timing.get("mask_prepare_ms", 0.0)),
                "depth_deproject_ms": float(extraction_timing.get("depth_deproject_ms", 0.0)),
                "depth_trim_ms": float(depth_trim_ms),
                "axis_template_fit_ms": 0.0,
                "endpoint_and_grasp_ms": 0.0,
                "quality_gate_ms": 0.0,
                "total_ms": float(total_ms),
            },
        }

    evaluation, axis_search_timing = _fit_axis(
        points,
        config,
        search_profile=search_profile,
        initial_axis=initial_axis,
    )
    fit_ms = float(axis_search_timing.get("total_ms", 0.0))
    pose_started = time.perf_counter()

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

    # Keep the original M37 point as a diagnostic reference.  It is not the
    # M37.1 grasp point because it lies on the near-opening rim and is selected
    # only by the highest projected image coordinate.
    legacy_rim_radius = 0.5 * (config.outer_radius_mm + config.inner_radius_mm)
    near_rim_top, near_rim_top_uv = _top_arc_point(
        near_center,
        axis_toward_camera,
        legacy_rim_radius,
        intrinsics,
        config.top_arc_sample_count,
    )
    grasp_circle_center = near_center - axis_toward_camera * config.grasp_axial_inset_mm
    crown_direction, crown_debug = _visible_crown_direction(
        points,
        evaluation.radial_inlier_mask,
        near_center,
        axis_toward_camera,
        grasp_circle_center,
        grasp_radius,
        intrinsics,
        config,
    )
    grasp_point = grasp_circle_center + grasp_radius * crown_direction
    grasp_point_uv_value = project_point(grasp_point, intrinsics)
    if grasp_point_uv_value is None:
        raise ValueError("near-side visible crown point cannot be projected")
    grasp_point_uv = (float(grasp_point_uv_value[0]), float(grasp_point_uv_value[1]))

    endpoint_and_grasp_ms = (time.perf_counter() - pose_started) * 1000.0
    quality_started = time.perf_counter()
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

    quality_gate_ms = (time.perf_counter() - quality_started) * 1000.0
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
        "search_profile_requested": str(search_profile),
        "search_profile_used": str(axis_search_timing.get("final_profile")),
        "accurate_fallback_used": bool(axis_search_timing.get("fallback_used", False)),
        "accurate_fallback_reasons": list(axis_search_timing.get("fallback_reasons") or []),
        "fast_acceptance_passed": bool(
            str(search_profile).strip().lower() == "fast"
            and len(rejection_reasons) == 0
            and float(evaluation.score) <= float(config.fast_accept_max_score)
        ),
        "fast_acceptance_reasons": list(
            _fast_fit_fallback_reasons(evaluation, config)
            if str(search_profile).strip().lower() == "fast" else []
        ),
        "warm_start_initial_axis": (
            np.asarray(initial_axis, dtype=np.float64).tolist()
            if initial_axis is not None else None
        ),
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
        "near_opening_rim_top_diagnostic": {
            "definition": "legacy_near_opening_wall_midline_highest_projected_point",
            "radius_mm": float(legacy_rim_radius),
            "point_camera_mm": near_rim_top.tolist(),
            "point_uv": [float(near_rim_top_uv[0]), float(near_rim_top_uv[1])],
        },
        "near_side_crown": {
            "definition": "camera_visible_cylindrical_arc_point_inset_from_near_opening",
            "radius_mm": float(grasp_radius),
            "radius_mode": str(config.grasp_radius_mode),
            "grasp_axial_inset_mm": float(config.grasp_axial_inset_mm),
            "grasp_circle_center_camera_mm": grasp_circle_center.tolist(),
            "radial_direction_camera": crown_direction.tolist(),
            "direction_source": str(crown_debug.get("direction_source")),
            "visible_arc": crown_debug,
            "grasp_point_camera_mm": grasp_point.tolist(),
            "grasp_point_uv": [float(grasp_point_uv[0]), float(grasp_point_uv[1])],
        },
        # Backward-compatible alias used by the M37 CSV/CLI and any early
        # experiments.  The grasp point now follows the M37.1 crown definition;
        # the old rim-top point is retained above for explicit diagnostics.
        "top_arc": {
            "definition": "M37.1_camera_visible_cylindrical_arc_point_inset_from_near_opening",
            "radius_mm": float(grasp_radius),
            "radius_mode": str(config.grasp_radius_mode),
            "near_rim_top_camera_mm": near_rim_top.tolist(),
            "near_rim_top_uv": [float(near_rim_top_uv[0]), float(near_rim_top_uv[1])],
            "grasp_axial_inset_mm": float(config.grasp_axial_inset_mm),
            "direction_source": str(crown_debug.get("direction_source")),
            "grasp_point_camera_mm": grasp_point.tolist(),
            "grasp_point_uv": [float(grasp_point_uv[0]), float(grasp_point_uv[1])],
        },
        "timing_ms": {
            "point_extraction_ms": float(extraction_ms),
            "mask_prepare_ms": float(extraction_timing.get("mask_prepare_ms", 0.0)),
            "depth_deproject_ms": float(extraction_timing.get("depth_deproject_ms", 0.0)),
            "depth_trim_ms": float(depth_trim_ms),
            "axis_template_fit_ms": float(fit_ms),
            "axis_search": axis_search_timing,
            "endpoint_and_grasp_ms": float(endpoint_and_grasp_ms),
            "quality_gate_ms": float(quality_gate_ms),
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
            "grasp_outer_circle_camera_mm": _circle_points(
                grasp_circle_center,
                axis_toward_camera,
                grasp_radius,
                96,
            ),
            "visible_crown_direction_camera": crown_direction,
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
