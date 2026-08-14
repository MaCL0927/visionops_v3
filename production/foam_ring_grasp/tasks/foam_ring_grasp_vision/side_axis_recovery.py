"""M39.4.0 pure side-lying foam-ring axis recovery.

Scope is deliberately narrow and diagnostic-only:
- no matched ``ring_mouth`` / no production grasp candidate;
- the ring is assumed to lie on the calibrated box floor (axis in box XY);
- recover an *undirected* cylinder axis from RGB silhouette + exact aligned depth;
- reconstruct the two nominal opening centres and select the preferred entry end;
- do NOT create a robot grasp candidate.  M39.4.1 will own insertion/collision routing.

The RGB mask only proposes the two orthogonal OBB directions.  Depth directional
anisotropy and a fixed-radius cylinder fit choose/refine the true cylinder axis.
This avoids treating perspective-visible inner wall pixels as a semantic mouth.
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
    _retain_components,
    _safe_float,
    _safe_int,
    _unit,
)
from .partial_opening_cylinder_m383 import _surface_depth_mode
from .segmentation import SegmentationInstance
from .side_surface_outer_contact_m385 import _evaluate_axis

_EPS = 1e-9


def _project(point: np.ndarray, intrinsics: Mapping[str, float]) -> np.ndarray:
    x, y, z = [float(v) for v in point]
    if z <= 1e-6:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)
    return np.asarray([
        float(intrinsics["fx"]) * x / z + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * y / z + float(intrinsics["cy"]),
    ], dtype=np.float64)


def _angle_0_180(vector_uv: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(vector_uv[1]), float(vector_uv[0]))) % 180.0)


def _undirected_angle_delta_deg(first: float, second: float) -> float:
    return abs(((float(first) - float(second) + 90.0) % 180.0) - 90.0)


def _obb_axis_seeds(mask: np.ndarray) -> List[Dict[str, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return []
    points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rect)
    rows: List[Dict[str, float]] = []
    for index in range(4):
        vector = corners[(index + 1) % 4] - corners[index]
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        angle = _angle_0_180(vector)
        if any(_undirected_angle_delta_deg(angle, row["angle_deg"]) < 1.0 for row in rows):
            continue
        rows.append({"angle_deg": angle, "edge_length_px": length})
    rows.sort(key=lambda row: row["angle_deg"])
    return rows[:2]


def _box_model(raw_config: Mapping[str, Any]) -> Optional[Dict[str, np.ndarray | float]]:
    section = raw_config.get("box_wall") or {}
    if not isinstance(section, Mapping):
        return None
    model = section.get("calibrated_model")
    if not isinstance(model, Mapping):
        return None
    try:
        origin = np.asarray(model["origin_camera_mm"], dtype=np.float64).reshape(3)
        axes = model["axes_camera"]
        x_axis = _unit(np.asarray(axes["x_right"], dtype=np.float64).reshape(3))
        y_axis = _unit(np.asarray(axes["y_down"], dtype=np.float64).reshape(3))
        z_axis = _unit(np.asarray(axes["z_inside"], dtype=np.float64).reshape(3))
        size = model["inner_size_mm"]
        width = float(size["width"])
        height = float(size["height"])
        depth = float(size["depth"])
    except Exception:
        return None
    return {
        "origin": origin,
        "x": x_axis,
        "y": y_axis,
        "z": z_axis,
        "width": width,
        "height": height,
        "depth": depth,
    }


def _to_box(point: np.ndarray, box: Mapping[str, Any]) -> np.ndarray:
    delta = np.asarray(point, dtype=np.float64) - np.asarray(box["origin"], dtype=np.float64)
    return np.asarray([
        float(np.dot(delta, np.asarray(box["x"], dtype=np.float64))),
        float(np.dot(delta, np.asarray(box["y"], dtype=np.float64))),
        float(np.dot(delta, np.asarray(box["z"], dtype=np.float64))),
    ], dtype=np.float64)


def _axis_box_angle_deg(axis: np.ndarray, box: Mapping[str, Any]) -> float:
    return float(math.degrees(math.atan2(
        float(np.dot(axis, np.asarray(box["y"], dtype=np.float64))),
        float(np.dot(axis, np.asarray(box["x"], dtype=np.float64))),
    )) % 180.0)


def _axis_from_box_angle(angle_deg: float, box: Mapping[str, Any]) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    return _unit(
        math.cos(theta) * np.asarray(box["x"], dtype=np.float64)
        + math.sin(theta) * np.asarray(box["y"], dtype=np.float64)
    )


def _floor_axis_from_image(
    image_angle_deg: float,
    anchor_camera_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    box: Mapping[str, Any],
) -> np.ndarray:
    target = np.asarray([
        math.cos(math.radians(image_angle_deg)),
        math.sin(math.radians(image_angle_deg)),
    ], dtype=np.float64)
    base_uv = _project(anchor_camera_mm, intrinsics)
    step_mm = 10.0
    jacobian = np.column_stack((
        (_project(anchor_camera_mm + step_mm * np.asarray(box["x"]), intrinsics) - base_uv) / step_mm,
        (_project(anchor_camera_mm + step_mm * np.asarray(box["y"]), intrinsics) - base_uv) / step_mm,
    ))
    coefficients, *_ = np.linalg.lstsq(jacobian, target, rcond=None)
    return _unit(
        float(coefficients[0]) * np.asarray(box["x"], dtype=np.float64)
        + float(coefficients[1]) * np.asarray(box["y"], dtype=np.float64)
    )


def _projected_axis_angle(axis: np.ndarray, anchor: np.ndarray, intrinsics: Mapping[str, float]) -> float:
    p0 = _project(anchor, intrinsics)
    p1 = _project(anchor + 20.0 * _unit(axis), intrinsics)
    return _angle_0_180(p1 - p0)


def _mask_extent_ratio(mask: np.ndarray, image_angle_deg: float) -> Tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    theta = math.radians(float(image_angle_deg))
    axis = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    along = points @ axis
    across = points @ perpendicular
    along_extent = float(np.percentile(along, 98) - np.percentile(along, 2))
    across_extent = float(np.percentile(across, 98) - np.percentile(across, 2))
    return float(along_extent / max(across_extent, 1e-6)), along_extent, across_extent


def _depth_gradients(depth_mm: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = depth_mm[mask]
    fill = float(np.median(values)) if len(values) else 0.0
    image = depth_mm.astype(np.float32).copy()
    image[~mask] = fill
    image = cv2.medianBlur(image, 5)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    metric_mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    return gx, gy, metric_mask


def _gradient_ratio(
    gx: np.ndarray,
    gy: np.ndarray,
    metric_mask: np.ndarray,
    image_angle_deg: float,
    stabilizer: float,
) -> Tuple[float, float, float]:
    theta = math.radians(float(image_angle_deg))
    cosine, sine = math.cos(theta), math.sin(theta)
    along = np.abs(gx * cosine + gy * sine)
    across = np.abs(-gx * sine + gy * cosine)
    if not np.any(metric_mask):
        return float("inf"), float("inf"), float("inf")
    along_median = float(np.median(along[metric_mask]))
    across_median = float(np.median(across[metric_mask]))
    return along_median, across_median, float(along_median / (across_median + max(1e-6, stabilizer)))


def _subsample_rows(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indexes = np.linspace(0, len(points) - 1, maximum).astype(np.int64)
    return points[indexes]


def _canonical_axis(axis: np.ndarray, center: np.ndarray, intrinsics: Mapping[str, float], vertical: bool) -> np.ndarray:
    axis = _unit(axis)
    delta = _project(center + 20.0 * axis, intrinsics) - _project(center, intrinsics)
    if vertical:
        if float(delta[1]) < 0.0:
            axis = -axis
    elif float(delta[0]) < 0.0:
        axis = -axis
    return axis


def _ray_wall_clearance_mm(point: np.ndarray, outward_axis: np.ndarray, box: Mapping[str, Any]) -> float:
    p = _to_box(point, box)
    limits = (float(box["width"]), float(box["height"]))
    # A reconstructed nominal endpoint can sit a few millimetres outside the
    # calibrated inner rectangle because of segmentation/depth noise.  That is
    # zero usable approach clearance, not an invitation to hit the orthogonal
    # wall hundreds of metres away through a near-zero direction component.
    if not (0.0 <= float(p[0]) <= limits[0] and 0.0 <= float(p[1]) <= limits[1]):
        return 0.0
    direction = np.asarray([
        float(np.dot(outward_axis, np.asarray(box["x"]))),
        float(np.dot(outward_axis, np.asarray(box["y"]))),
    ], dtype=np.float64)
    candidates: List[float] = []
    for index in range(2):
        component = float(direction[index])
        if abs(component) <= 1e-9:
            continue
        target = limits[index] if component > 0.0 else 0.0
        distance = (target - float(p[index])) / component
        if distance >= 0.0:
            candidates.append(float(distance))
    return float(min(candidates)) if candidates else float("inf")


def _json_vector(value: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]




def _mouth_axis_ratio_from_mask(mask: np.ndarray) -> Optional[float]:
    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    try:
        _center, axes, _angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    major = max(float(axes[0]), float(axes[1]))
    minor = min(float(axes[0]), float(axes[1]))
    if major <= 1e-6:
        return None
    return float(minor / major)


def apply_m39401_mouth_topology_arbitration(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reclassify very flat semantic mouths as side-view pseudo mouths.

    The segmentation class remains untouched.  This gate only decides whether a
    matched semantic ``ring_mouth`` is geometrically valid for the frozen M39.3
    front/tilted production branch.  A very flat projected ellipse is treated as
    perspective-visible inner wall / side-view opening evidence and is rerouted
    to M39.4.0 side-axis recovery.
    """

    section = raw_config.get("m39_4_0_side_axis_recovery") or {}
    if not isinstance(section, Mapping):
        section = {}
    gate = section.get("mouth_topology_gate") or {}
    if not isinstance(gate, Mapping):
        gate = {}
    enabled = bool(gate.get("enabled", True))
    threshold = _safe_float(gate.get("side_view_axis_ratio_max"), 0.50)
    threshold = float(np.clip(threshold, 0.05, 0.95))
    by_id = {int(item.instance_id): item for item in instances}
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "M39.4.0.1_side_topology_arbitration",
        "enabled": enabled,
        "side_view_axis_ratio_max": threshold,
        "evaluated_pair_count": 0,
        "pseudo_mouth_pair_count": 0,
        "pseudo_mouth_ring_ids": [],
        "pseudo_mouth_ids": [],
        "pairs": [],
        "visible_mouth_candidate_invalidated": False,
        "status": "disabled" if not enabled else "ok",
    }
    if not enabled:
        scene["m39_4_0_1_mouth_topology_arbitration"] = summary
        return summary

    pseudo_ring_ids: List[int] = []
    pseudo_mouth_ids: List[int] = []
    for item in scene.get("instances") or []:
        if not isinstance(item, dict) or str(item.get("pose_strategy") or "") != "m38_1_front_annulus":
            continue
        ring_id = item.get("ring_instance_id")
        mouth_id = item.get("mouth_instance_id")
        if ring_id is None or mouth_id is None:
            continue
        branch = item.get("m38_branch_a") if isinstance(item.get("m38_branch_a"), dict) else {}
        item["m38_branch_a"] = branch
        ellipse_quality = branch.get("ellipse_quality") if isinstance(branch.get("ellipse_quality"), Mapping) else {}
        ratio = ellipse_quality.get("minor_major_ratio")
        source = "m38_branch_a.ellipse_quality"
        try:
            ratio_value = float(ratio) if ratio is not None else None
        except (TypeError, ValueError):
            ratio_value = None
        if ratio_value is None or not math.isfinite(ratio_value):
            mouth = by_id.get(int(mouth_id))
            ratio_value = _mouth_axis_ratio_from_mask(mouth.mask) if mouth is not None else None
            source = "mouth_mask_fitEllipse"

        pseudo = bool(ratio_value is not None and ratio_value <= threshold)
        topology = "SIDE_VIEW_PSEUDO_MOUTH" if pseudo else "FRONT_MOUTH_CANDIDATE"
        row = {
            "ring_instance_id": int(ring_id),
            "mouth_instance_id": int(mouth_id),
            "mouth_axis_ratio": float(ratio_value) if ratio_value is not None else None,
            "axis_ratio_source": source,
            "threshold": threshold,
            "topology": topology,
            "visible_mouth_production_allowed": not pseudo,
            "forced_side_axis_recovery": pseudo,
        }
        branch["m39_4_0_1_mouth_topology"] = dict(row)
        summary["pairs"].append(row)
        summary["evaluated_pair_count"] += 1
        if pseudo:
            pseudo_ring_ids.append(int(ring_id))
            pseudo_mouth_ids.append(int(mouth_id))

    pseudo_ring_ids = sorted(set(pseudo_ring_ids))
    pseudo_mouth_ids = sorted(set(pseudo_mouth_ids))
    summary["pseudo_mouth_ring_ids"] = pseudo_ring_ids
    summary["pseudo_mouth_ids"] = pseudo_mouth_ids
    summary["pseudo_mouth_pair_count"] = len(pseudo_ring_ids)
    summary["status"] = "pseudo_mouth_reroute_required" if pseudo_ring_ids else "no_pseudo_mouth"

    candidate = scene.get("robot_candidate")
    candidate_ring_id = None
    if isinstance(candidate, Mapping):
        target = candidate.get("target") if isinstance(candidate.get("target"), Mapping) else {}
        candidate_ring_id = target.get("ring_instance_id")
    try:
        candidate_is_pseudo = candidate_ring_id is not None and int(candidate_ring_id) in pseudo_ring_ids
    except (TypeError, ValueError):
        candidate_is_pseudo = False
    if candidate_is_pseudo:
        summary["visible_mouth_candidate_invalidated"] = True
        summary["invalidated_ring_instance_id"] = int(candidate_ring_id)
        summary["invalidated_grasp_branch"] = str(candidate.get("grasp_branch") or scene.get("selected_grasp_branch") or "")
        scene["robot_candidate"] = None
        scene["selected_grasp_branch"] = "m39_4_0_1_pseudo_mouth_reroute_pending"

    scene["m39_4_0_1_mouth_topology_arbitration"] = summary
    return summary

def recover_pure_side_axis(
    ring: SegmentationInstance,
    all_rings: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recover one pure side-lying cylinder axis without creating a grasp pose.

    Selection is intentionally two-stage for speed:
    1. evaluate the two RGB OBB directions with depth-gradient anisotropy plus
       the known 70/85 axial/diameter silhouette prior;
    2. run only a tiny fixed-radius cylinder refinement around the winning seed.

    Perspective-visible inner-wall pixels therefore cannot decide the opening
    direction by themselves, and the expensive cylinder fit is not run over a
    large 3-D orientation search.
    """

    started = time.perf_counter()
    section = raw_config.get("m39_4_0_side_axis_recovery") or {}
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
        "status": "axis_uncertain",
        "reliable": False,
        "rejection_reasons": [],
        "warnings": [],
    }
    reasons: List[str] = result["rejection_reasons"]
    warnings: List[str] = result["warnings"]
    box = _box_model(raw_config)
    if box is None:
        reasons.append("m3940_calibrated_box_model_unavailable")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result

    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    outer_radius = 0.5 * _safe_float(object_cfg.get("nominal_outer_diameter_mm"), 85.0)
    axial_length = _safe_float(object_cfg.get("axial_length_mm"), 70.0)
    expected_ratio = axial_length / max(2.0 * outer_radius, 1e-6)

    prepare_started = time.perf_counter()
    mask = _erode(ring.mask, _safe_int(section.get("ring_mask_erode_px"), 2))
    other_mask = np.zeros_like(mask, dtype=bool)
    for other in all_rings:
        if int(other.instance_id) != int(ring.instance_id):
            other_mask |= other.mask.astype(bool)
    mask &= ~_dilate(other_mask, _safe_int(section.get("neighbor_exclusion_dilate_px"), 1))
    mask &= (depth_mm >= minimum_depth) & (depth_mm <= maximum_depth)
    surface_depth = _surface_depth_mode(depth_mm, mask)
    result["surface_depth_mm"] = float(surface_depth) if surface_depth is not None else None
    if surface_depth is None:
        reasons.append("m3940_surface_depth_unavailable")
        result["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return result

    mask &= (
        (depth_mm >= float(surface_depth) - _safe_float(section.get("surface_depth_front_tolerance_mm"), 18.0))
        & (depth_mm <= float(surface_depth) + _safe_float(section.get("surface_depth_back_tolerance_mm"), 55.0))
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
    result["surface_mask_pixel_count"] = int(np.count_nonzero(mask))
    result["surface_component_count"] = int(component_count)
    result["surface_kept_component_count"] = int(kept_component_count)
    result.setdefault("timing_ms", {})["surface_prepare_ms"] = (time.perf_counter() - prepare_started) * 1000.0

    points, pixels = _deproject_mask(depth_mm, mask, intrinsics, minimum_depth, maximum_depth)
    result["surface_point_count"] = int(len(points))
    if len(points) < _safe_int(section.get("minimum_side_points"), 150):
        reasons.append("m3940_insufficient_side_surface_points")
    if len(points) < 3:
        result["timing_ms"]["total_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    anchor = np.median(points, axis=0)
    seeds = _obb_axis_seeds(ring.mask)
    result["rgb_axis_seeds"] = seeds
    if len(seeds) != 2:
        reasons.append("m3940_rgb_obb_axis_seeds_unavailable")
        result["timing_ms"]["total_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    # Fast two-hypothesis decision.  For a side cylinder, depth changes mainly
    # across the axis, not along it.  The physical 70/85 aspect ratio provides a
    # weak RGB prior that resolves near-symmetric depth-gradient cases.
    quick_started = time.perf_counter()
    gx, gy, gradient_mask = _depth_gradients(depth_mm, mask)
    gradient_weight = _safe_float(section.get("depth_gradient_weight"), 10.0)
    silhouette_weight = _safe_float(section.get("silhouette_ratio_weight"), 12.0)
    stabilizer = _safe_float(section.get("gradient_stabilizer_mm_per_px"), 0.5)
    quick_rows: List[Dict[str, Any]] = []
    for seed_index, seed_row in enumerate(seeds):
        image_angle = float(seed_row["angle_deg"])
        silhouette_ratio, along_extent, across_extent = _mask_extent_ratio(ring.mask, image_angle)
        gradient_along, gradient_across, gradient_ratio = _gradient_ratio(
            gx, gy, gradient_mask, image_angle, stabilizer
        )
        quick_score = (
            gradient_weight * float(gradient_ratio)
            + silhouette_weight * abs(float(silhouette_ratio) - expected_ratio)
        )
        quick_rows.append({
            "seed_index": int(seed_index),
            "seed_image_angle_deg": float(image_angle),
            "edge_length_px": float(seed_row["edge_length_px"]),
            "quick_score": float(quick_score),
            "gradient_along_median_mm_per_px": float(gradient_along),
            "gradient_across_median_mm_per_px": float(gradient_across),
            "depth_gradient_axis_ratio": float(gradient_ratio),
            "silhouette_axial_to_diameter_ratio": float(silhouette_ratio),
            "silhouette_expected_ratio": float(expected_ratio),
            "silhouette_along_extent_px": float(along_extent),
            "silhouette_across_extent_px": float(across_extent),
        })
    quick_rows.sort(key=lambda row: float(row["quick_score"]))
    quick_margin = float(quick_rows[1]["quick_score"] - quick_rows[0]["quick_score"])
    result["axis_seed_candidates"] = quick_rows
    result["axis_score_margin"] = quick_margin
    result["timing_ms"]["two_axis_quick_score_ms"] = (time.perf_counter() - quick_started) * 1000.0

    winning_seed = quick_rows[0]
    seed_axis = _floor_axis_from_image(
        float(winning_seed["seed_image_angle_deg"]), anchor, intrinsics, box
    )
    seed_box_angle = _axis_box_angle_deg(seed_axis, box)

    # Refine the quick-score winner.  When the cheap A/B discriminator is
    # close, refine BOTH orthogonal seeds and let fixed-radius 3-D geometry
    # decide.  This is the M39.4.0.1 rescue for silhouettes where perspective
    # makes the cheap 70/85 prior prefer the wrong OBB direction.
    refine_started = time.perf_counter()
    offsets = section.get("local_refine_offsets_deg", [-2, 0, 2])
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
        offsets = [-2, 0, 2]
    central_fraction = float(np.clip(_safe_float(section.get("central_axis_fraction"), 0.60), 0.30, 0.90))
    tail = 0.5 * (1.0 - central_fraction)
    maximum_fit_points = max(100, _safe_int(section.get("maximum_fit_points"), 350))
    offset_weight = _safe_float(section.get("refine_offset_weight_per_deg"), 0.08)
    dual_threshold = _safe_float(section.get("dual_seed_refine_quick_margin_below"), 1.20)
    dual_seed_refine = bool(quick_margin < dual_threshold)
    seeds_to_refine = list(quick_rows) if dual_seed_refine else [quick_rows[0]]
    refinements: List[Dict[str, Any]] = []
    for seed_candidate in seeds_to_refine:
        seed_axis = _floor_axis_from_image(
            float(seed_candidate["seed_image_angle_deg"]), anchor, intrinsics, box
        )
        seed_box_angle = _axis_box_angle_deg(seed_axis, box)
        for offset in [float(v) for v in offsets]:
            axis = _axis_from_box_angle(seed_box_angle + offset, box)
            image_angle = _projected_axis_angle(axis, anchor, intrinsics)
            axial = points @ axis
            low, high = np.quantile(axial, [tail, 1.0 - tail])
            fit_points = _subsample_rows(points[(axial >= low) & (axial <= high)], maximum_fit_points)
            if len(fit_points) < 3:
                continue
            cylinder = _evaluate_axis(
                fit_points,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                axis,
                outer_radius,
                _safe_int(section.get("fixed_radius_iterations"), 3),
                _safe_float(section.get("radial_inlier_threshold_mm"), 7.0),
            )
            silhouette_ratio, along_extent, across_extent = _mask_extent_ratio(ring.mask, image_angle)
            gradient_along, gradient_across, gradient_ratio = _gradient_ratio(
                gx, gy, gradient_mask, image_angle, stabilizer
            )
            geometry_score = float(cylinder["score"]) + offset_weight * abs(offset)
            quick_prior_score = (
                gradient_weight * float(gradient_ratio)
                + silhouette_weight * abs(float(silhouette_ratio) - expected_ratio)
            )
            # Keep the old combined score for diagnostics, but M39.4.0.1 uses
            # full cylinder geometry as the final discriminator once refinement
            # is running.
            total_score = geometry_score + quick_prior_score
            refinements.append({
                "seed_index": int(seed_candidate["seed_index"]),
                "seed_image_angle_deg": float(seed_candidate["seed_image_angle_deg"]),
                "seed_quick_score": float(seed_candidate["quick_score"]),
                "refine_offset_deg": float(offset),
                "axis_box_angle_deg": _axis_box_angle_deg(axis, box),
                "axis_image_angle_deg": float(image_angle),
                "axis_camera_undirected": _json_vector(axis),
                "cylinder_score": float(cylinder["score"]),
                "full_geometry_score": float(geometry_score),
                "quick_prior_score": float(quick_prior_score),
                "total_score": float(total_score),
                "radial_residual_median_mm": float(cylinder["radial_residual_median_mm"]),
                "radial_residual_p90_mm": float(cylinder["radial_residual_p90_mm"]),
                "radial_inlier_ratio": float(cylinder["radial_inlier_ratio"]),
                "depth_gradient_axis_ratio": float(gradient_ratio),
                "silhouette_axial_to_diameter_ratio": float(silhouette_ratio),
                "_axis": axis,
                "_cylinder": cylinder,
            })
    if not refinements:
        reasons.append("m39401_local_cylinder_refinement_failed")
        result["timing_ms"]["local_cylinder_refine_ms"] = (time.perf_counter() - refine_started) * 1000.0
        result["timing_ms"]["total_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    selection_key = "full_geometry_score" if dual_seed_refine else "total_score"
    refinements.sort(key=lambda row: float(row[selection_key]))
    seed_best_rows: List[Dict[str, Any]] = []
    for seed_id in sorted({int(row["seed_index"]) for row in refinements}):
        candidates = [row for row in refinements if int(row["seed_index"]) == seed_id]
        if candidates:
            seed_best_rows.append(min(candidates, key=lambda row: float(row["full_geometry_score"])))
    seed_best_rows.sort(key=lambda row: float(row["full_geometry_score"]))
    full_geometry_margin = None
    if len(seed_best_rows) >= 2:
        full_geometry_margin = float(seed_best_rows[1]["full_geometry_score"] - seed_best_rows[0]["full_geometry_score"])

    best = refinements[0]
    cylinder = best.pop("_cylinder")
    axis = np.asarray(best.pop("_axis"), dtype=np.float64)
    result["dual_seed_refine_applied"] = bool(dual_seed_refine)
    result["dual_seed_refine_quick_margin_threshold"] = float(dual_threshold)
    result["full_geometry_score_margin"] = full_geometry_margin
    result["full_geometry_seed_best"] = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in seed_best_rows
    ]
    result["selected_axis_refinement"] = dict(best)
    result["axis_refinement_candidates"] = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in refinements
    ]
    result["timing_ms"]["local_cylinder_refine_ms"] = (time.perf_counter() - refine_started) * 1000.0

    center = np.asarray(cylinder["axis_point"], dtype=np.float64)
    preliminary_angle = float(best["axis_image_angle_deg"])
    vertical_threshold = _safe_float(section.get("vertical_axis_threshold_deg_from_vertical"), 20.0)
    vertical_case = bool(abs(90.0 - preliminary_angle) <= vertical_threshold)
    axis = _canonical_axis(axis, center, intrinsics, vertical_case)
    image_angle = _projected_axis_angle(axis, center, intrinsics)
    if image_angle >= 179.5:
        image_angle = 0.0

    half_length = 0.5 * axial_length
    endpoint_a = center - half_length * axis
    endpoint_b = center + half_length * axis
    uv_a = _project(endpoint_a, intrinsics)
    uv_b = _project(endpoint_b, intrinsics)
    clearance_a = _ray_wall_clearance_mm(endpoint_a, -axis, box)
    clearance_b = _ray_wall_clearance_mm(endpoint_b, axis, box)
    if vertical_case:
        selected_label = "B" if clearance_b >= clearance_a else "A"
        selection_rule = "vertical_choose_larger_box_wall_clearance"
    else:
        selected_label = "B" if float(uv_b[0]) >= float(uv_a[0]) else "A"
        selection_rule = "nonvertical_choose_image_right_endpoint"
    selected_point = endpoint_b if selected_label == "B" else endpoint_a
    selected_uv = uv_b if selected_label == "B" else uv_a
    selected_clearance = clearance_b if selected_label == "B" else clearance_a

    center_box = _to_box(center, box)
    expected_center_z = float(box["depth"]) - outer_radius
    center_height_error = abs(float(center_box[2]) - expected_center_z)
    floor_error_deg = math.degrees(math.asin(float(np.clip(abs(np.dot(axis, np.asarray(box["z"]))), 0.0, 1.0))))

    if dual_seed_refine and full_geometry_margin is not None:
        minimum_full_margin = _safe_float(section.get("minimum_full_geometry_score_margin"), 0.80)
        if float(full_geometry_margin) < minimum_full_margin:
            reasons.append("m39401_full_geometry_axis_margin_too_small")
    if float(best["radial_residual_median_mm"]) > _safe_float(section.get("maximum_radial_residual_median_mm"), 4.0):
        reasons.append("m39401_radial_residual_median_too_large")
    if float(best["radial_residual_p90_mm"]) > _safe_float(section.get("maximum_radial_residual_p90_mm"), 8.0):
        reasons.append("m39401_radial_residual_p90_too_large")
    if float(best["radial_inlier_ratio"]) < _safe_float(section.get("minimum_radial_inlier_ratio"), 0.85):
        reasons.append("m39401_radial_inlier_ratio_too_small")

    center_height_warning = _safe_float(section.get("center_height_warning_mm"), 20.0)
    center_reliable = bool(center_height_error <= center_height_warning)
    if not center_reliable:
        warnings.append("m39401_center_height_unreliable_axis_kept")

    axis_reliable = len(reasons) == 0
    result.update({
        "status": "axis_recovered" if axis_reliable else "axis_uncertain",
        "reliable": bool(axis_reliable),
        "axis_reliable": bool(axis_reliable),
        "center_reliable": bool(center_reliable),
        "center_status": "RELIABLE" if center_reliable else "UNRELIABLE_WARNING_ONLY",
        "axis_camera_undirected": _json_vector(axis),
        "axis_box_angle_deg": _axis_box_angle_deg(axis, box),
        "axis_image_angle_deg_0_180": float(image_angle),
        "axis_floor_plane_error_deg": float(floor_error_deg),
        "center_camera_mm": _json_vector(center),
        "center_box_mm": _json_vector(center_box),
        "expected_floor_resting_center_z_box_mm": float(expected_center_z),
        "center_height_error_mm": float(center_height_error),
        "nominal_outer_radius_mm": float(outer_radius),
        "nominal_axial_length_mm": float(axial_length),
        "vertical_case": bool(vertical_case),
        "vertical_deviation_deg": float(abs(90.0 - image_angle)),
        "vertical_axis_threshold_deg_from_vertical": float(vertical_threshold),
        "endpoint_a_camera_mm": _json_vector(endpoint_a),
        "endpoint_b_camera_mm": _json_vector(endpoint_b),
        "endpoint_a_uv": _json_vector(uv_a),
        "endpoint_b_uv": _json_vector(uv_b),
        "endpoint_a_outward_wall_clearance_mm": float(clearance_a),
        "endpoint_b_outward_wall_clearance_mm": float(clearance_b),
        "entry_endpoint": selected_label,
        "entry_selection_rule": selection_rule,
        "entry_center_camera_mm": _json_vector(selected_point),
        "entry_center_uv": _json_vector(selected_uv),
        "entry_outward_wall_clearance_mm": float(selected_clearance),
        "robot_ready": False,
        "robot_routing_enabled": False,
    })
    result["timing_ms"]["total_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def attach_m3940_side_axis_recovery(
    scene: Dict[str, Any],
    instances: Sequence[SegmentationInstance],
    depth_mm: np.ndarray,
    intrinsics: Mapping[str, float],
    raw_config: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    section = raw_config.get("m39_4_0_side_axis_recovery") or {}
    if not isinstance(section, Mapping):
        section = {}
    summary: Dict[str, Any] = {
        "schema_version": "1.1",
        "stage": "M39.4.0.1_side_lying_axis_recovery",
        "enabled": bool(section.get("enabled", True)),
        "mode": str(section.get("mode") or "online_diagnostic_only"),
        "robot_routing_enabled": bool(section.get("robot_routing_enabled", False)),
        "executed": False,
        "status": "disabled" if not bool(section.get("enabled", True)) else "not_applicable",
        "robot_ready": False,
        "fits": [],
    }
    if not summary["enabled"]:
        scene["m39_4_0_side_axis_recovery"] = summary
        return summary
    if bool(section.get("only_when_no_robot_candidate", True)) and isinstance(scene.get("robot_candidate"), Mapping):
        summary["status"] = "not_needed_visible_mouth_candidate_available"
        scene["m39_4_0_side_axis_recovery"] = summary
        return summary

    unmatched_ids = {int(v) for v in (scene.get("unmatched_ring_ids") or [])}
    topology = scene.get("m39_4_0_1_mouth_topology_arbitration") or {}
    pseudo_ids = {
        int(v) for v in (topology.get("pseudo_mouth_ring_ids") or [])
    } if isinstance(topology, Mapping) else set()
    side_candidate_ids = unmatched_ids | pseudo_ids
    all_foam_rings = [item for item in instances if item.class_name == "foam_ring"]
    rings = [item for item in all_foam_rings if int(item.instance_id) in side_candidate_ids]
    summary["unmatched_ring_ids"] = sorted(unmatched_ids)
    summary["pseudo_mouth_ring_ids"] = sorted(pseudo_ids)
    summary["side_candidate_ring_ids"] = sorted(side_candidate_ids)
    if not rings:
        summary["status"] = "no_side_axis_candidate"
        scene["m39_4_0_side_axis_recovery"] = summary
        return summary

    depth_rows = {
        int(row.get("ring_instance_id")): row
        for row in ((scene.get("depth_layering") or {}).get("candidates") or [])
        if isinstance(row, Mapping) and row.get("ring_instance_id") is not None
    }
    rings.sort(key=lambda ring: (
        _safe_int((depth_rows.get(int(ring.instance_id)) or {}).get("depth_layer_index"), 999),
        _safe_float((depth_rows.get(int(ring.instance_id)) or {}).get("surface_depth_mm"), float("inf")),
        -float(ring.confidence),
    ))
    maximum = max(1, _safe_int(section.get("maximum_rings_to_evaluate"), 3))
    pseudo_rows = {}
    if isinstance(topology, Mapping):
        for row in topology.get("pairs") or []:
            if isinstance(row, Mapping) and row.get("ring_instance_id") is not None:
                pseudo_rows[int(row["ring_instance_id"])] = row
    fits: List[Dict[str, Any]] = []
    for ring in rings[:maximum]:
        fit = recover_pure_side_axis(ring, all_foam_rings, depth_mm, intrinsics, raw_config)
        rid = int(ring.instance_id)
        if rid in pseudo_ids:
            fit["side_candidate_source"] = "PSEUDO_MOUTH_REROUTE"
            row = pseudo_rows.get(rid) or {}
            fit["pseudo_mouth_axis_ratio"] = row.get("mouth_axis_ratio")
            fit["semantic_mouth_instance_id"] = row.get("mouth_instance_id")
        else:
            fit["side_candidate_source"] = "NO_MOUTH"
        fits.append(fit)

    reliable = [row for row in fits if bool(row.get("reliable", False))]
    selected = reliable[0] if reliable else None
    summary.update({
        "executed": True,
        "evaluated_ring_count": int(len(fits)),
        "reliable_ring_count": int(len(reliable)),
        "fits": fits,
        "selected_ring_instance_id": selected.get("ring_instance_id") if selected else None,
        "selected": selected,
        "robot_ready": False,
        "terminal_reject": True,
    })
    if selected is not None:
        summary.update({
            "status": "axis_recovered_validation_only",
            "selected_grasp_branch": "m39_4_0_1_side_axis_recovery",
            "reason": "m39401_axis_recovered_validation_only",
            "display_reason_short": "M39.4.0.1 AXIS RECOVERED - NO ROBOT MOTION",
            "display_reason_detail": "Side topology/axis recovered; validate before M39.4.1 insertion routing",
            "operator_action": "validate_m39_4_0_axis_then_implement_m39_4_1",
        })
        scene["selected_grasp_branch"] = "m39_4_0_1_side_axis_recovery"
        scene["selected_ring_instance_id"] = int(selected["ring_instance_id"])
    else:
        summary.update({
            "status": "axis_uncertain",
            "selected_grasp_branch": "m39_4_0_1_side_axis_uncertain",
            "reason": "m39401_side_axis_uncertain",
            "display_reason_short": "REJECT: M39.4.0.1 SIDE AXIS UNCERTAIN",
            "display_reason_detail": "No robot motion; collect/debug the side-lying sample",
            "operator_action": "inspect_m39_4_0_axis_debug",
        })
        scene["selected_grasp_branch"] = "m39_4_0_1_side_axis_uncertain"
    scene["operator_action"] = summary["operator_action"]
    summary["timing_ms"] = {"total_ms": (time.perf_counter() - started) * 1000.0}
    scene["m39_4_0_side_axis_recovery"] = summary
    return summary


def draw_m3940_side_axis_overlay(
    image_bgr: np.ndarray,
    summary: Mapping[str, Any],
) -> np.ndarray:
    output = image_bgr.copy()
    selected = summary.get("selected") if isinstance(summary.get("selected"), Mapping) else None
    if selected is None:
        fits = summary.get("fits") if isinstance(summary.get("fits"), list) else []
        selected = fits[0] if fits and isinstance(fits[0], Mapping) else None
    if not isinstance(selected, Mapping):
        return output
    a = selected.get("endpoint_a_uv")
    b = selected.get("endpoint_b_uv")
    if not (isinstance(a, list) and len(a) == 2 and isinstance(b, list) and len(b) == 2):
        return output
    pa = tuple(int(round(float(v))) for v in a)
    pb = tuple(int(round(float(v))) for v in b)
    reliable = bool(selected.get("reliable", False))
    line_color = (0, 255, 255) if reliable else (0, 128, 255)
    cv2.line(output, pa, pb, line_color, 3, cv2.LINE_AA)
    cv2.circle(output, pa, 6, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(output, pb, 6, (255, 180, 0), 2, cv2.LINE_AA)
    entry = str(selected.get("entry_endpoint") or "")
    chosen = pb if entry == "B" else pa
    cv2.circle(output, chosen, 10, (0, 255, 0) if reliable else (0, 0, 255), 3, cv2.LINE_AA)
    angle = selected.get("axis_image_angle_deg_0_180")
    margin = selected.get("axis_score_margin")
    source = str(selected.get("side_candidate_source") or "")
    source_short = " PSEUDO" if source == "PSEUDO_MOUTH_REROUTE" else ""
    text = f"M39.4.0.1 SIDE{source_short} axis={float(angle):.1f}deg entry={entry}" if angle is not None else "M39.4.0.1 SIDE"
    cv2.putText(output, text, (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)
    if margin is not None:
        cv2.putText(output, f"score_margin={float(margin):.2f} robot=OFF", (14, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return output
