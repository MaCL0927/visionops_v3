"""M35.2 offline geometry with complete pre-grasp motion collision checks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .segmentation import SegmentationInstance
from .box_model_3d import (
    BoxModel3D,
    box_model_from_dict,
    box_projection,
    check_swept_prism_against_box,
    combine_collision_checks,
    validate_model_for_capture,
)
from .gripper_model_3d import (
    check_full_gripper_pregrasp_motion,
    check_full_gripper_static_final_pose,
)


@dataclass(frozen=True)
class GeometryConfig:
    raw: Dict[str, Any]

    def section(self, name: str) -> Dict[str, Any]:
        value = self.raw.get(name) or {}
        return dict(value) if isinstance(value, dict) else {}


@dataclass
class PlaneModel:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    inlier_mask: np.ndarray
    inlier_ratio: float
    residual_median_mm: float
    residual_p95_mm: float


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


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _optimization_settings(config: GeometryConfig) -> Dict[str, Any]:
    """Resolve online geometry evaluation settings through M36.4.2.

    The historical/offline default remains ``exhaustive`` when the section is
    absent.  ``staged`` preserves M36.4.1 global candidate budgeting, while
    ``first_valid`` processes pre-ranked pairs lazily and exits as soon as one
    complete collision-checked grasp is found.
    """
    section = config.section("geometry_optimization")
    enabled = bool(section.get("enabled", False))
    mode = str(section.get("mode") or ("staged" if enabled else "exhaustive")).strip().lower()
    if mode not in {"first_valid", "staged", "exhaustive"}:
        mode = "exhaustive"
    initial_budget = max(1, _safe_int(section.get("initial_full_candidate_budget"), 4))
    maximum_budget = max(initial_budget, _safe_int(section.get("maximum_full_candidate_budget"), 8))
    minimum_valid = max(1, _safe_int(section.get("minimum_valid_full_candidates"), 2))
    return {
        "enabled": enabled,
        "mode": mode,
        "timing_enabled": bool(section.get("timing_enabled", True)),
        "skip_rejected_pairs": bool(section.get("skip_rejected_pairs", True)),
        "prefer_top_layer": bool(section.get("prefer_top_layer", True)),
        "round_robin_pairs": bool(section.get("round_robin_pairs", True)),
        "cache_neighbor_point_clouds": bool(section.get("cache_neighbor_point_clouds", True)),
        "initial_full_candidate_budget": initial_budget,
        "maximum_full_candidate_budget": maximum_budget,
        "minimum_valid_full_candidates": minimum_valid,
        "expand_if_no_valid": bool(section.get("expand_if_no_valid", True)),
        "stop_after_first_valid_target": bool(section.get("stop_after_first_valid_target", True)),
        "stop_after_first_valid_candidate": bool(section.get("stop_after_first_valid_candidate", True)),
        "maximum_pairs_to_fully_analyze": max(
            1, _safe_int(section.get("maximum_pairs_to_fully_analyze"), 3)
        ),
        "maximum_full_candidates_per_pair": max(
            1, _safe_int(section.get("maximum_full_candidates_per_pair"), 12)
        ),
    }


def _clock_search_settings(config: GeometryConfig) -> Dict[str, Any]:
    section = config.section("clock_search")
    primary_raw = section.get("primary_clock_hours")
    primary_hours: List[int] = []
    if isinstance(primary_raw, Sequence) and not isinstance(primary_raw, (str, bytes)):
        for value in primary_raw:
            hour = _safe_int(value, -1)
            if hour == 0:
                hour = 12
            if 1 <= hour <= 12 and hour not in primary_hours:
                primary_hours.append(hour)
    if not primary_hours:
        # Four cardinal directions plus four evenly distributed diagonals.
        # The remaining 1/4/7/10 o'clock directions form the fallback batch.
        primary_hours = [12, 2, 3, 5, 6, 8, 9, 11]
    return {
        "mode": str(section.get("mode") or "adaptive_8_plus_4").strip().lower(),
        "primary_clock_hours": primary_hours,
        "fallback_to_remaining": bool(section.get("fallback_to_remaining", True)),
    }


def _adaptive_clock_batches(config: GeometryConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    settings = _clock_search_settings(config)
    base = _clock_positions(12)
    by_hour = {int(row["clock_hour"]): dict(row) for row in base}
    primary: List[Dict[str, Any]] = []
    for order, hour in enumerate(settings["primary_clock_hours"]):
        row = by_hour.get(int(hour))
        if row is None:
            continue
        row = dict(row)
        row["search_batch"] = "primary"
        row["search_order"] = int(order)
        primary.append(row)
    primary_indexes = {int(row["clock_index"]) for row in primary}
    fallback: List[Dict[str, Any]] = []
    if settings["fallback_to_remaining"]:
        for row in base:
            if int(row["clock_index"]) in primary_indexes:
                continue
            item = dict(row)
            item["search_batch"] = "fallback"
            item["search_order"] = int(len(primary) + len(fallback))
            fallback.append(item)
    return primary, fallback


def _pair_preselection_metrics(
    ring: SegmentationInstance,
    mouth: SegmentationInstance,
    association: Mapping[str, Any],
    depth: np.ndarray,
    config: GeometryConfig,
) -> Dict[str, Any]:
    """Cheap pair ranking without point-cloud construction or plane fitting."""
    section = config.section("pair_preselection")
    stride = max(1, _safe_int(section.get("depth_sample_stride"), 4))
    minimum_depth = _safe_float(config.section("depth").get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(config.section("depth").get("maximum_mm"), 3000.0)
    ring_erode = max(0, _safe_int(section.get("ring_erode_px"), 1))
    mouth_exclusion = max(0, _safe_int(section.get("mouth_exclusion_px"), 2))
    front_mask = _erode(ring.mask, ring_erode) & ~_dilate(mouth.mask, mouth_exclusion)
    sampled_mask = front_mask[::stride, ::stride]
    sampled_depth = depth[::stride, ::stride]
    values = sampled_depth[sampled_mask]
    valid = values[(values >= minimum_depth) & (values <= maximum_depth)]
    valid_count = int(valid.size)
    sampled_count = int(values.size)
    valid_ratio = float(valid_count) / float(max(1, sampled_count))
    median_mm = float(np.median(valid)) if valid_count else None
    p25_mm = float(np.percentile(valid, 25)) if valid_count else None
    confidence = 0.5 * (float(ring.confidence) + float(mouth.confidence))
    return {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "sparse_front_depth_median_mm": median_mm,
        "sparse_front_depth_p25_mm": p25_mm,
        "sparse_depth_valid_count": valid_count,
        "sparse_depth_sample_count": sampled_count,
        "sparse_depth_valid_ratio": valid_ratio,
        "association_score": _safe_float(association.get("association_score"), 0.0),
        "containment": _safe_float(association.get("containment"), 0.0),
        "segmentation_confidence": float(confidence),
    }


def _pair_preselection_rank(metrics: Mapping[str, Any], prefer_top_layer: bool) -> Tuple[Any, ...]:
    median = metrics.get("sparse_front_depth_median_mm")
    has_depth = median is not None and math.isfinite(float(median))
    return (
        bool(has_depth),
        -float(median) if has_depth and prefer_top_layer else 0.0,
        float(metrics.get("sparse_depth_valid_ratio") or 0.0),
        float(metrics.get("association_score") or 0.0),
        float(metrics.get("segmentation_confidence") or 0.0),
        float(metrics.get("containment") or 0.0),
    )


def _deferred_pair_result(
    ring: SegmentationInstance,
    mouth: SegmentationInstance,
    association: Mapping[str, Any],
    preselection: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "ring_confidence": float(ring.confidence),
        "mouth_confidence": float(mouth.confidence),
        "association": dict(association),
        "pair_preselection": dict(preselection),
        "processing_status": "deferred",
        "deferred_reason": str(reason),
        "eligible": False,
        "robot_ready": False,
        "warnings": [],
        "rejection_reasons": [],
        "grasp": {
            "mode": "rim_pinch",
            "clock_candidates": [],
            "best_clock_candidate": None,
            "best_light_clock_candidate": None,
        },
        "candidate_evaluation": {
            "mode": "first_valid",
            "status": "deferred",
            "full_evaluated_count": 0,
            "full_valid_count": 0,
            "deferred_count": 12,
        },
        "timing_ms": {"total_ms": 0.0},
    }


def _aggregate_timing_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        timing = row.get("timing_ms") if isinstance(row.get("timing_ms"), Mapping) else {}
        for key, value in timing.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            values.setdefault(str(key), []).append(number)
    return {
        key: {
            "count": float(len(items)),
            "total_ms": float(sum(items)),
            "mean_ms": float(sum(items) / len(items)),
            "max_ms": float(max(items)),
        }
        for key, items in values.items()
        if items
    }


def _kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return cv2.erode(mask.astype(np.uint8), _kernel(radius), iterations=1).astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius), iterations=1).astype(bool)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if abs(moments["m00"]) < 1e-9:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return (0.0, 0.0)
        return (float(xs.mean()), float(ys.mean()))
    return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))


def point_in_mask(mask: np.ndarray, uv: Tuple[float, float]) -> bool:
    u, v = uv
    x = int(round(u))
    y = int(round(v))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])



def filled_outer_envelope(
    mask: np.ndarray,
    dilate_px: int = 0,
    close_px: int = 0,
    minimum_component_area_ratio: float = 0.03,
) -> np.ndarray:
    """Build a filled outer envelope from all meaningful mask components.

    PT/RKNN masks can split one physical ring into a front-rim component and a
    cylindrical-side component. Filling only the largest contour then loses the
    mouth. M34_new keeps all meaningful components, closes small gaps, and fills
    their convex hull.
    """
    binary = mask.astype(np.uint8)
    if close_px > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(close_px), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    envelope = np.zeros_like(binary)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest = float(np.max(areas)) if areas.size else 0.0
        minimum_area = max(8.0, largest * max(0.0, float(minimum_component_area_ratio)))
        points: List[np.ndarray] = []
        for label_id in range(1, count):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            ys, xs = np.nonzero(labels == label_id)
            if xs.size:
                points.append(np.column_stack((xs, ys)).astype(np.int32))
        if points:
            hull = cv2.convexHull(np.concatenate(points, axis=0).reshape(-1, 1, 2))
            cv2.fillConvexPoly(envelope, hull, 1)
    if dilate_px > 0:
        envelope = cv2.dilate(envelope, _kernel(dilate_px), iterations=1)
    return envelope.astype(bool)


def _expanded_bbox_mask(shape: Tuple[int, int], bbox: Tuple[int, int, int, int], expand_px: int) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(math.floor(x1)) - expand_px)
    y1 = max(0, int(math.floor(y1)) - expand_px)
    x2 = min(width, int(math.ceil(x2)) + expand_px)
    y2 = min(height, int(math.ceil(y2)) + expand_px)
    mask = np.zeros(shape, dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _associate_ring_mouths_detailed(
    rings: Sequence[SegmentationInstance],
    mouths: Sequence[SegmentationInstance],
    config: GeometryConfig,
) -> Tuple[
    List[Tuple[SegmentationInstance, SegmentationInstance, Dict[str, Any]]],
    List[SegmentationInstance],
    List[SegmentationInstance],
    List[Dict[str, Any]],
]:
    section = config.section("association")
    use_envelope = bool(section.get("use_filled_outer_envelope", True))
    envelope_dilate_px = _safe_int(section.get("envelope_dilate_px"), 4)
    envelope_close_px = _safe_int(section.get("envelope_close_px"), 3)
    component_ratio = _safe_float(section.get("minimum_component_area_ratio"), 0.03)
    min_containment = _safe_float(
        section.get("minimum_envelope_containment", section.get("minimum_containment")),
        0.55 if use_envelope else 0.72,
    )
    min_ratio = _safe_float(section.get("minimum_mouth_to_ring_area_ratio"), 0.025)
    max_ratio = _safe_float(section.get("maximum_mouth_to_ring_area_ratio"), 0.70)
    require_center = bool(section.get("minimum_center_inside", True))
    max_center_distance = _safe_float(section.get("maximum_normalized_center_distance"), 0.55)

    fallback_enabled = bool(section.get("fallback_bbox_enabled", True))
    fallback_expand_px = _safe_int(section.get("fallback_bbox_expand_px"), 10)
    fallback_min_containment = _safe_float(section.get("fallback_minimum_containment"), 0.30)
    fallback_max_center_distance = _safe_float(section.get("fallback_maximum_normalized_center_distance"), 0.70)
    fallback_score_penalty = _safe_float(section.get("fallback_score_penalty"), 0.30)

    envelopes = [
        filled_outer_envelope(
            ring.mask,
            envelope_dilate_px,
            close_px=envelope_close_px,
            minimum_component_area_ratio=component_ratio,
        )
        if use_envelope
        else ring.mask.astype(bool)
        for ring in rings
    ]

    candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
    debug_rows: List[Dict[str, Any]] = []
    for mouth_index, mouth in enumerate(mouths):
        mouth_area = max(1, mouth.area_px)
        mouth_center = mouth.centroid_uv
        best_debug: Optional[Dict[str, Any]] = None
        for ring_index, ring in enumerate(rings):
            envelope = envelopes[ring_index]
            envelope_area = max(1, int(np.count_nonzero(envelope)))
            raw_overlap = int(np.count_nonzero(mouth.mask & ring.mask))
            envelope_overlap = int(np.count_nonzero(mouth.mask & envelope))
            raw_containment = float(raw_overlap) / float(mouth_area)
            containment = float(envelope_overlap) / float(mouth_area)
            area_ratio = float(mouth_area) / float(envelope_area)
            center_inside = point_in_mask(envelope, mouth_center)
            ring_center = mask_centroid(envelope)
            center_distance = math.hypot(mouth_center[0] - ring_center[0], mouth_center[1] - ring_center[1])
            ys, xs = np.nonzero(envelope)
            if xs.size:
                diagonal = max(1.0, math.hypot(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
            else:
                x1, y1, x2, y2 = ring.bbox_xyxy
                diagonal = max(1.0, math.hypot(x2 - x1, y2 - y1))
            normalized_distance = center_distance / diagonal

            fallback_mask = _expanded_bbox_mask(envelope.shape, ring.bbox_xyxy, fallback_expand_px)
            fallback_overlap = int(np.count_nonzero(mouth.mask & fallback_mask))
            fallback_containment = float(fallback_overlap) / float(mouth_area)
            fallback_center_inside = point_in_mask(fallback_mask, mouth_center)

            strict_ok = (
                containment >= min_containment
                and min_ratio <= area_ratio <= max_ratio
                and (center_inside or not require_center)
                and normalized_distance <= max_center_distance
            )
            fallback_ok = (
                fallback_enabled
                and not strict_ok
                and fallback_containment >= fallback_min_containment
                and min_ratio <= area_ratio <= max_ratio
                and fallback_center_inside
                and normalized_distance <= fallback_max_center_distance
            )

            mode = "strict_envelope" if strict_ok else ("bbox_fallback" if fallback_ok else "rejected")
            base_score = containment - 0.22 * normalized_distance - 0.08 * abs(area_ratio - 0.28)
            score = base_score if strict_ok else (
                fallback_containment - 0.22 * normalized_distance - 0.08 * abs(area_ratio - 0.28) - fallback_score_penalty
            )
            metrics: Dict[str, Any] = {
                "raw_mask_containment": raw_containment,
                "envelope_containment": containment,
                "fallback_bbox_containment": fallback_containment,
                "containment": containment if strict_ok else fallback_containment,
                "mouth_to_ring_area_ratio": area_ratio,
                "center_distance_px": center_distance,
                "normalized_center_distance": normalized_distance,
                "center_inside_envelope": bool(center_inside),
                "center_inside_fallback_bbox": bool(fallback_center_inside),
                "association_score": score,
                "association_mode": mode,
                "used_filled_outer_envelope": bool(use_envelope),
            }
            candidate_debug = {
                "mouth_instance_id": int(mouth.instance_id),
                "ring_instance_id": int(ring.instance_id),
                "accepted": bool(strict_ok or fallback_ok),
                **metrics,
            }
            if (
                best_debug is None
                or (bool(candidate_debug["accepted"]) and not bool(best_debug.get("accepted")))
                or (
                    bool(candidate_debug["accepted"]) == bool(best_debug.get("accepted"))
                    and float(candidate_debug["association_score"]) > float(best_debug["association_score"])
                )
            ):
                best_debug = candidate_debug
            if strict_ok or fallback_ok:
                candidates.append((score, ring_index, mouth_index, metrics))
        if best_debug is not None:
            debug_rows.append(best_debug)
        else:
            debug_rows.append(
                {
                    "mouth_instance_id": int(mouth.instance_id),
                    "ring_instance_id": None,
                    "accepted": False,
                    "association_mode": "no_ring_candidates",
                    "association_score": None,
                }
            )

    candidates.sort(key=lambda row: row[0], reverse=True)
    used_rings = set()
    used_mouths = set()
    matches: List[Tuple[SegmentationInstance, SegmentationInstance, Dict[str, Any]]] = []
    for _, ring_index, mouth_index, metrics in candidates:
        if ring_index in used_rings or mouth_index in used_mouths:
            continue
        used_rings.add(ring_index)
        used_mouths.add(mouth_index)
        matches.append((rings[ring_index], mouths[mouth_index], metrics))

    matched_pairs = {(int(ring.instance_id), int(mouth.instance_id)) for ring, mouth, _ in matches}
    for row in debug_rows:
        row["matched"] = (
            row.get("ring_instance_id") is not None
            and (int(row["ring_instance_id"]), int(row["mouth_instance_id"])) in matched_pairs
        )
        if not row["matched"]:
            if not row.get("accepted"):
                row["rejection_reason"] = "association_thresholds_not_met"
            else:
                row["rejection_reason"] = "one_to_one_assignment_conflict"
        else:
            row["rejection_reason"] = None

    unmatched_rings = [ring for index, ring in enumerate(rings) if index not in used_rings]
    unmatched_mouths = [mouth for index, mouth in enumerate(mouths) if index not in used_mouths]
    return matches, unmatched_rings, unmatched_mouths, debug_rows


def associate_ring_mouths(
    rings: Sequence[SegmentationInstance],
    mouths: Sequence[SegmentationInstance],
    config: GeometryConfig,
) -> Tuple[List[Tuple[SegmentationInstance, SegmentationInstance, Dict[str, Any]]], List[SegmentationInstance]]:
    """Backward-compatible public pairing API."""
    matches, unmatched_rings, _, _ = _associate_ring_mouths_detailed(rings, mouths, config)
    return matches, unmatched_rings

def depth_pixels_to_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: Mapping[str, float],
    minimum_mm: float,
    maximum_mm: float,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & (depth >= minimum_mm) & (depth <= maximum_mm)
    if stride > 1:
        sample = np.zeros_like(valid)
        sample[::stride, ::stride] = True
        valid &= sample
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    z = depth[ys, xs].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    pixels = np.column_stack((xs, ys)).astype(np.int32)
    return points, pixels


def _plane_from_three(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-7:
        return None
    normal = normal / norm
    return normal, -float(np.dot(normal, p1))


def _refine_plane(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    offset = -float(np.dot(normal, centroid))
    return normal, offset, centroid


def fit_plane_ransac(points: np.ndarray, config: GeometryConfig) -> Optional[PlaneModel]:
    section = config.section("plane")
    iterations = _safe_int(section.get("ransac_iterations"), 320)
    threshold = _safe_float(section.get("inlier_threshold_mm"), 5.0)
    seed = _safe_int(section.get("random_seed"), 3401)
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask = None
    best_count = -1
    best_median = float("inf")
    for _ in range(max(1, iterations)):
        indexes = rng.choice(len(points), size=3, replace=False)
        candidate = _plane_from_three(points[indexes])
        if candidate is None:
            continue
        normal, offset = candidate
        residuals = np.abs(points @ normal + offset)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residuals[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_mask = inliers
    if best_mask is None or int(np.count_nonzero(best_mask)) < 3:
        return None
    inlier_points = points[best_mask]
    normal, offset, centroid = _refine_plane(inlier_points)
    # Keep the normal facing the camera origin.  The actual insertion/approach
    # direction is therefore -normal (from the front face into the ring).
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal
        offset = -offset
    residuals = np.abs(points @ normal + offset)
    final_mask = residuals <= threshold
    final_residuals = residuals[final_mask]
    return PlaneModel(
        normal=normal,
        offset=float(offset),
        centroid=points[final_mask].mean(axis=0),
        inlier_mask=final_mask,
        inlier_ratio=float(np.count_nonzero(final_mask)) / float(len(points)),
        residual_median_mm=float(np.median(final_residuals)) if len(final_residuals) else float("inf"),
        residual_p95_mm=float(np.percentile(final_residuals, 95)) if len(final_residuals) else float("inf"),
    )


def ray_plane_intersection(
    uv: Tuple[float, float],
    intrinsics: Mapping[str, float],
    plane: PlaneModel,
) -> Optional[np.ndarray]:
    u, v = uv
    ray = np.asarray(
        [
            (float(u) - float(intrinsics["cx"])) / float(intrinsics["fx"]),
            (float(v) - float(intrinsics["cy"])) / float(intrinsics["fy"]),
            1.0,
        ],
        dtype=np.float64,
    )
    denominator = float(np.dot(plane.normal, ray))
    if abs(denominator) < 1e-8:
        return None
    scale = -float(plane.offset) / denominator
    if scale <= 0.0:
        return None
    return ray * scale


def project_point(point: np.ndarray, intrinsics: Mapping[str, float]) -> Optional[Tuple[float, float]]:
    if point.shape != (3,) or point[2] <= 1e-6:
        return None
    u = float(intrinsics["fx"]) * float(point[0]) / float(point[2]) + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * float(point[1]) / float(point[2]) + float(intrinsics["cy"])
    return (u, v)


def fit_mouth_ellipse(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (center_u, center_v), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if axis_a >= axis_b:
        major_px, minor_px, major_angle_deg = float(axis_a), float(axis_b), float(angle)
    else:
        major_px, minor_px, major_angle_deg = float(axis_b), float(axis_a), float(angle + 90.0)
    major_angle_deg %= 180.0
    return {
        "center_uv": (float(center_u), float(center_v)),
        "major_px": major_px,
        "minor_px": minor_px,
        "angle_deg": major_angle_deg,
        "contour": contour.reshape(-1, 2),
    }


def ellipse_axis_endpoints(ellipse: Mapping[str, Any], major: bool = True) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    center_u, center_v = ellipse["center_uv"]
    angle_deg = float(ellipse["angle_deg"]) + (0.0 if major else 90.0)
    diameter = float(ellipse["major_px"] if major else ellipse["minor_px"])
    radius = diameter / 2.0
    angle = math.radians(angle_deg)
    delta = np.asarray([math.cos(angle) * radius, math.sin(angle) * radius], dtype=np.float64)
    center = np.asarray([center_u, center_v], dtype=np.float64)
    first = center - delta
    second = center + delta
    return (tuple(first.tolist()), tuple(second.tolist()))


def _axis_metric_on_plane(
    ellipse: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    plane: PlaneModel,
    major: bool,
) -> Optional[Dict[str, Any]]:
    endpoint_a_uv, endpoint_b_uv = ellipse_axis_endpoints(ellipse, major=major)
    endpoint_a = ray_plane_intersection(endpoint_a_uv, intrinsics, plane)
    endpoint_b = ray_plane_intersection(endpoint_b_uv, intrinsics, plane)
    if endpoint_a is None or endpoint_b is None:
        return None
    vector = endpoint_b - endpoint_a
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return None
    return {
        "length_mm": length,
        "endpoint_a_uv": endpoint_a_uv,
        "endpoint_b_uv": endpoint_b_uv,
        "endpoint_a_camera_mm": endpoint_a,
        "endpoint_b_camera_mm": endpoint_b,
        "axis_camera": vector / length,
    }



def _ellipse_normal_candidates(
    ellipse: Mapping[str, Any],
    intrinsics: Mapping[str, float],
) -> Tuple[float, List[np.ndarray]]:
    major = max(1e-6, float(ellipse["major_px"]))
    minor = max(1e-6, float(ellipse["minor_px"]))
    ratio = float(np.clip(minor / major, 0.0, 1.0))
    tilt_rad = math.acos(ratio)
    minor_angle = math.radians((float(ellipse["angle_deg"]) + 90.0) % 180.0)
    direction_xy = np.asarray(
        [
            math.cos(minor_angle) / float(intrinsics["fx"]),
            math.sin(minor_angle) / float(intrinsics["fy"]),
        ],
        dtype=np.float64,
    )
    direction_xy /= max(float(np.linalg.norm(direction_xy)), 1e-12)
    xy_magnitude = math.sin(tilt_rad)
    z = -math.cos(tilt_rad)
    first = np.asarray([xy_magnitude * direction_xy[0], xy_magnitude * direction_xy[1], z], dtype=np.float64)
    second = np.asarray([-xy_magnitude * direction_xy[0], -xy_magnitude * direction_xy[1], z], dtype=np.float64)
    return math.degrees(tilt_rad), [first, second]


def _project_points_to_uv(points: np.ndarray, intrinsics: Mapping[str, float]) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    z = np.maximum(points[:, 2], 1e-9)
    u = float(intrinsics["fx"]) * points[:, 0] / z + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * points[:, 1] / z + float(intrinsics["cy"])
    return np.column_stack((u, v))


def _ellipse_boundary_points(
    points: np.ndarray,
    ellipse: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    inner_ratio: float,
    outer_ratio: float,
    minimum_points: int,
) -> np.ndarray:
    if len(points) == 0:
        return points
    uv = _project_points_to_uv(points, intrinsics)
    center = np.asarray(ellipse["center_uv"], dtype=np.float64)
    angle = math.radians(float(ellipse["angle_deg"]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    delta = uv - center[None, :]
    major_coordinate = delta[:, 0] * cosine + delta[:, 1] * sine
    minor_coordinate = -delta[:, 0] * sine + delta[:, 1] * cosine
    normalized_radius = np.sqrt(
        np.square(major_coordinate / max(1e-6, float(ellipse["major_px"]) * 0.5))
        + np.square(minor_coordinate / max(1e-6, float(ellipse["minor_px"]) * 0.5))
    )
    keep = (normalized_radius >= inner_ratio) & (normalized_radius <= outer_ratio)
    if int(np.count_nonzero(keep)) >= minimum_points:
        return points[keep]
    return points


def _fixed_normal_plane(
    points: np.ndarray,
    normal: np.ndarray,
    threshold_mm: float,
) -> PlaneModel:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] > 0.0:
        normal = -normal
    offsets = -(points @ normal)
    offset = float(np.median(offsets))
    residuals = np.abs(points @ normal + offset)
    inliers = residuals <= threshold_mm
    if int(np.count_nonzero(inliers)) < 3:
        inliers = np.ones(len(points), dtype=bool)
    inlier_points = points[inliers]
    centroid = inlier_points.mean(axis=0)
    inlier_residuals = residuals[inliers]
    return PlaneModel(
        normal=normal,
        offset=offset,
        centroid=centroid,
        inlier_mask=inliers,
        inlier_ratio=float(np.count_nonzero(inliers)) / float(max(1, len(points))),
        residual_median_mm=float(np.median(inlier_residuals)) if len(inlier_residuals) else float("inf"),
        residual_p95_mm=float(np.percentile(inlier_residuals, 95)) if len(inlier_residuals) else float("inf"),
    )


def build_pose_plane(
    ellipse: Mapping[str, Any],
    intrinsics: Mapping[str, float],
    front_points: np.ndarray,
    depth_plane: PlaneModel,
    config: GeometryConfig,
) -> Tuple[PlaneModel, Dict[str, Any]]:
    """Stabilize the ring-axis pose with the projected mouth ellipse.

    The foam rim is rounded, so the largest RANSAC plane can be a tangent patch
    or cylindrical side wall.  The mouth ellipse gives a much more repeatable
    tilt magnitude.  Depth points are retained to choose the tilt direction and
    anchor the mouth plane in 3D.
    """
    pose_cfg = config.section("pose")
    mode = str(pose_cfg.get("normal_mode") or "auto").strip().lower()
    ellipse_tilt_deg, candidates = _ellipse_normal_candidates(ellipse, intrinsics)
    depth_tilt_deg = math.degrees(math.acos(float(np.clip(abs(depth_plane.normal[2]), 0.0, 1.0))))
    candidate_disagreements = [
        math.degrees(math.acos(float(np.clip(np.dot(candidate, depth_plane.normal), -1.0, 1.0))))
        for candidate in candidates
    ]
    minimum_disagreement = min(candidate_disagreements) if candidate_disagreements else 180.0

    near_frontal_override = (
        mode == "auto"
        and bool(pose_cfg.get("near_frontal_depth_override_enabled", True))
        and ellipse_tilt_deg <= _safe_float(pose_cfg.get("near_frontal_max_ellipse_tilt_deg"), 28.0)
        and depth_tilt_deg <= _safe_float(pose_cfg.get("near_frontal_max_depth_tilt_deg"), 20.0)
        and depth_plane.inlier_ratio >= _safe_float(pose_cfg.get("near_frontal_min_depth_inlier_ratio"), 0.25)
        and depth_plane.residual_p95_mm <= _safe_float(pose_cfg.get("near_frontal_max_depth_residual_p95_mm"), 8.0)
    )
    trust_depth = (
        mode == "depth_plane"
        or near_frontal_override
        or (
            mode == "auto"
            and depth_plane.inlier_ratio >= _safe_float(pose_cfg.get("trust_depth_plane_min_inlier_ratio"), 0.82)
            and depth_plane.residual_p95_mm <= _safe_float(pose_cfg.get("trust_depth_plane_max_residual_p95_mm"), 4.0)
            and minimum_disagreement <= _safe_float(pose_cfg.get("trust_depth_plane_max_disagreement_deg"), 12.0)
        )
    )
    if trust_depth:
        diagnostics = {
            "normal_source": "near_frontal_depth_plane" if near_frontal_override else "depth_plane",
            "depth_plane_tilt_deg": float(depth_tilt_deg),
            "ellipse_tilt_deg": float(ellipse_tilt_deg),
            "normal_disagreement_deg": float(minimum_disagreement),
            "boundary_support_points": int(len(front_points)),
        }
        return depth_plane, diagnostics

    ambiguous_below = _safe_float(pose_cfg.get("ellipse_azimuth_ambiguous_below_deg"), 8.0)
    threshold = _safe_float(pose_cfg.get("stabilized_inlier_threshold_mm"), 8.0)
    inner_ratio = _safe_float(pose_cfg.get("boundary_inner_ratio"), 0.95)
    outer_ratio = _safe_float(pose_cfg.get("boundary_outer_ratio"), 1.35)
    minimum_points = _safe_int(pose_cfg.get("minimum_boundary_support_points"), 40)
    support = _ellipse_boundary_points(
        front_points,
        ellipse,
        intrinsics,
        inner_ratio,
        outer_ratio,
        minimum_points,
    )

    if ellipse_tilt_deg < ambiguous_below:
        selected_normal = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        pose_plane = _fixed_normal_plane(support, selected_normal, threshold)
        sign_method = "camera_axis_for_near_circular_mouth"
    else:
        scored: List[Tuple[Tuple[float, float, float], PlaneModel]] = []
        raw_direction_weight = _safe_float(pose_cfg.get("depth_direction_tiebreak_weight"), 0.15)
        for candidate in candidates:
            candidate_plane = _fixed_normal_plane(support, candidate, threshold)
            residuals = np.abs(support @ candidate_plane.normal + candidate_plane.offset)
            median = float(np.median(residuals)) if len(residuals) else float("inf")
            p75 = float(np.percentile(residuals, 75)) if len(residuals) else float("inf")
            direction_penalty = raw_direction_weight * math.degrees(
                math.acos(float(np.clip(np.dot(candidate_plane.normal, depth_plane.normal), -1.0, 1.0)))
            )
            scored.append(((median + direction_penalty, p75, direction_penalty), candidate_plane))
        scored.sort(key=lambda item: item[0])
        pose_plane = scored[0][1]
        sign_method = "ellipse_minor_axis_depth_residual"

    final_disagreement = math.degrees(
        math.acos(float(np.clip(np.dot(pose_plane.normal, depth_plane.normal), -1.0, 1.0)))
    )
    diagnostics = {
        "normal_source": "ellipse_stabilized",
        "sign_method": sign_method,
        "depth_plane_tilt_deg": float(depth_tilt_deg),
        "ellipse_tilt_deg": float(ellipse_tilt_deg),
        "normal_disagreement_deg": float(final_disagreement),
        "boundary_support_points": int(len(support)),
        "stabilized_residual_median_mm": float(pose_plane.residual_median_mm),
        "stabilized_residual_p95_mm": float(pose_plane.residual_p95_mm),
    }
    return pose_plane, diagnostics


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("零向量无法归一化")
    return value / norm


def _json_vector(vector: Optional[np.ndarray]) -> Optional[List[float]]:
    if vector is None:
        return None
    return [float(value) for value in np.asarray(vector).reshape(-1).tolist()]


def _json_uv(value: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _clock_positions(count: int) -> List[Dict[str, Any]]:
    """Return fixed image-clock positions.

    12 o'clock is image-up. Hours increase clockwise, so 3 o'clock is image
    right, 6 is down and 9 is left.
    """
    count = max(4, int(count))
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        clock_angle = 360.0 * float(index) / float(count)
        image_angle = clock_angle - 90.0
        hour = 12 if index == 0 else int(round(12.0 * index / count))
        if count == 12:
            hour = 12 if index == 0 else index
        rows.append(
            {
                "clock_index": index,
                "clock_hour": hour,
                "clock_angle_deg_cw_from_12": clock_angle,
                "image_angle_deg_from_positive_x": image_angle,
            }
        )
    return rows


def _ray_point(center_uv: Tuple[float, float], angle_deg: float, distance_px: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    return np.asarray(center_uv, dtype=np.float64) + np.asarray(
        [math.cos(angle), math.sin(angle)], dtype=np.float64
    ) * float(distance_px)


def _mask_value(mask: np.ndarray, point: np.ndarray) -> bool:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _mouth_boundary_on_ray(
    mouth_mask: np.ndarray,
    center_uv: Tuple[float, float],
    angle_deg: float,
    step_px: float = 0.35,
) -> Optional[Dict[str, Any]]:
    max_distance = float(math.hypot(mouth_mask.shape[1], mouth_mask.shape[0]))
    entered = False
    last_inside: Optional[np.ndarray] = None
    last_distance = 0.0
    for distance in np.arange(0.0, max_distance, step_px):
        point = _ray_point(center_uv, angle_deg, float(distance))
        inside = _mask_value(mouth_mask, point)
        if inside:
            entered = True
            last_inside = point
            last_distance = float(distance)
        elif entered:
            break
    if last_inside is None:
        return None
    return {
        "uv": (float(last_inside[0]), float(last_inside[1])),
        "distance_px": float(last_distance),
    }


def _outer_boundary_on_ray(
    ring_mask: np.ndarray,
    center_uv: Tuple[float, float],
    angle_deg: float,
    start_distance_px: float,
    maximum_search_px: float,
    maximum_gap_px: int,
    step_px: float = 0.35,
) -> Optional[Dict[str, Any]]:
    """Find the first local rim-material run after the mouth boundary.

    The search is physically capped. If the visible side wall continues beyond
    the cap, the result is marked ambiguous instead of inventing a wall width.
    """
    maximum_search_px = max(step_px, float(maximum_search_px))
    gap_limit_steps = max(1, int(math.ceil(maximum_gap_px / step_px)))
    entered = False
    outside_steps = 0
    last_inside: Optional[np.ndarray] = None
    last_distance = float(start_distance_px)
    for delta in np.arange(step_px, maximum_search_px + step_px, step_px):
        distance = float(start_distance_px + delta)
        point = _ray_point(center_uv, angle_deg, distance)
        inside = _mask_value(ring_mask, point)
        if inside:
            entered = True
            outside_steps = 0
            last_inside = point
            last_distance = distance
        elif entered:
            outside_steps += 1
            if outside_steps > gap_limit_steps:
                break
    if last_inside is None:
        return None
    reached_cap = (last_distance - start_distance_px) >= maximum_search_px - 2.0 * step_px
    return {
        "uv": (float(last_inside[0]), float(last_inside[1])),
        "distance_px": float(last_distance),
        "wall_run_px": float(last_distance - start_distance_px),
        "ambiguous": bool(reached_cap),
    }


def _project_polygon(points_camera: Sequence[np.ndarray], intrinsics: Mapping[str, float]) -> Optional[np.ndarray]:
    projected: List[Tuple[float, float]] = []
    for point in points_camera:
        uv = project_point(np.asarray(point, dtype=np.float64), intrinsics)
        if uv is None:
            return None
        projected.append(uv)
    if len(projected) < 3:
        return None
    return cv2.convexHull(np.rint(np.asarray(projected, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2))


def _finger_cross_section_corners(
    center: np.ndarray,
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    thickness_mm: float,
    width_mm: float,
) -> List[np.ndarray]:
    half_t = float(thickness_mm) * 0.5
    half_w = float(width_mm) * 0.5
    return [
        center + closing_axis * sx * half_t + tangent_axis * sy * half_w
        for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0))
    ]


def _finger_sweep_polygon(
    front_center: np.ndarray,
    final_center: np.ndarray,
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    thickness_mm: float,
    width_mm: float,
    intrinsics: Mapping[str, float],
) -> Optional[np.ndarray]:
    corners = _finger_cross_section_corners(
        front_center,
        closing_axis,
        tangent_axis,
        thickness_mm,
        width_mm,
    ) + _finger_cross_section_corners(
        final_center,
        closing_axis,
        tangent_axis,
        thickness_mm,
        width_mm,
    )
    return _project_polygon(corners, intrinsics)


def _polygon_mask(shape: Tuple[int, int], polygon: Optional[np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon is not None and len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon.reshape(-1, 2), 1)
    return mask.astype(bool)


def _polygon_border_margin_px(polygon: Optional[np.ndarray], shape: Tuple[int, int]) -> float:
    if polygon is None or len(polygon) == 0:
        return -1.0
    points = polygon.reshape(-1, 2).astype(np.float64)
    height, width = shape
    margins = np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            float(width - 1) - points[:, 0],
            float(height - 1) - points[:, 1],
        )
    )
    return float(np.min(margins))




def _box_inner_polygon_uv(shape: Tuple[int, int], config: GeometryConfig) -> Optional[np.ndarray]:
    """Resolve the configured inner usable box opening in image coordinates."""
    section = config.section("box_wall")
    if not bool(section.get("enabled", False)):
        return None
    values = section.get("inner_polygon_normalized")
    if not isinstance(values, list) or len(values) < 3:
        return None
    height, width = shape
    points: List[List[float]] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0]) * float(max(1, width - 1))
            y = float(item[1]) * float(max(1, height - 1))
        except (TypeError, ValueError):
            continue
        points.append([x, y])
    if len(points) < 3:
        return None
    return np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _box_wall_model(
    shape: Tuple[int, int],
    config: GeometryConfig,
    intrinsics: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    section = config.section("box_wall")
    model_name = str(section.get("model") or "disabled")
    if not bool(section.get("enabled", False)):
        return {"enabled": False, "model": model_name, "status": "disabled"}
    if model_name == "calibrated_3d_cuboid":
        model = _resolve_box_model_3d(config)
        if model is None:
            return {
                "enabled": True,
                "model": model_name,
                "status": "unconfigured",
                "calibration_file": section.get("_resolved_calibration_file") or section.get("calibration_file"),
            }
        validation = validate_model_for_capture(
            model,
            shape,
            intrinsics or {},
            resolution_tolerance_px=_safe_int(section.get("resolution_tolerance_px"), 0),
            intrinsics_relative_tolerance=_safe_float(section.get("intrinsics_relative_tolerance"), 0.02),
        )
        projection = box_projection(model, intrinsics or {}) if intrinsics else {}
        return {
            "enabled": True,
            "model": model_name,
            "status": "ready" if validation.get("valid") else "capture_mismatch",
            "calibration_file": model.source_path,
            "coordinate_frame": model.camera_frame_id,
            "origin_camera_mm": model.origin_camera_mm.astype(float).tolist(),
            "rotation_camera_from_box_rows": model.rotation_camera_from_box.astype(float).tolist(),
            "inner_size_mm": {
                "width": float(model.inner_size_mm[0]),
                "height": float(model.inner_size_mm[1]),
                "depth": float(model.inner_size_mm[2]),
            },
            "safety_margin_mm": dict(model.safety_margin_mm),
            "validation": validation,
            **projection,
        }

    # Legacy 2-D opening polygon remains only as an explicit fallback mode.
    polygon = _box_inner_polygon_uv(shape, config)
    enabled = polygon is not None
    return {
        "enabled": bool(enabled),
        "model": model_name,
        "status": "ready" if enabled else "unconfigured",
        "inner_polygon_uv": polygon.reshape(-1, 2).astype(float).tolist() if polygon is not None else None,
        "minimum_wall_clearance_mm": _safe_float(section.get("minimum_wall_clearance_mm"), 8.0),
        "minimum_wall_clearance_px": _safe_float(section.get("minimum_wall_clearance_px"), 0.0),
        "minimum_outer_finger_containment": _safe_float(section.get("minimum_outer_finger_containment"), 0.985),
        "minimum_inner_finger_containment": _safe_float(section.get("minimum_inner_finger_containment"), 0.950),
        "hard_reject": bool(section.get("hard_reject", True)),
    }

def _check_box_wall_clearance(
    inner_sweep: np.ndarray,
    outer_sweep: np.ndarray,
    mm_per_px: float,
    config: GeometryConfig,
) -> Dict[str, Any]:
    """Conservative stage-1 collision check against a calibrated box opening.

    The configured polygon describes the inner usable box opening. A finger
    insertion sweep must remain inside it. Distance-transform clearance is
    converted to millimetres with the local ring scale.
    """
    section = config.section("box_wall")
    polygon = _box_inner_polygon_uv(inner_sweep.shape, config)
    if not bool(section.get("enabled", False)):
        return {
            "enabled": False,
            "status": "disabled",
            "inner_polygon_uv": None,
            "inner_finger_containment": 1.0,
            "outer_finger_containment": 1.0,
            "clearance_px": None,
            "clearance_mm": None,
            "required_clearance_mm": _safe_float(section.get("minimum_wall_clearance_mm"), 8.0),
            "required_clearance_px": _safe_float(section.get("minimum_wall_clearance_px"), 0.0),
            "hard_reject": False,
        }
    if polygon is None:
        return {
            "enabled": True,
            "status": "unconfigured",
            "inner_polygon_uv": None,
            "inner_finger_containment": 0.0,
            "outer_finger_containment": 0.0,
            "clearance_px": None,
            "clearance_mm": None,
            "required_clearance_mm": _safe_float(section.get("minimum_wall_clearance_mm"), 8.0),
            "required_clearance_px": _safe_float(section.get("minimum_wall_clearance_px"), 0.0),
            "hard_reject": bool(section.get("hard_reject", True)),
        }

    usable = np.zeros(inner_sweep.shape, dtype=np.uint8)
    cv2.fillPoly(usable, [polygon], 1)
    usable_bool = usable.astype(bool)
    inner_area = max(1, int(np.count_nonzero(inner_sweep)))
    outer_area = max(1, int(np.count_nonzero(outer_sweep)))
    inner_containment = float(np.count_nonzero(inner_sweep & usable_bool)) / float(inner_area)
    outer_containment = float(np.count_nonzero(outer_sweep & usable_bool)) / float(outer_area)

    check_inner = bool(section.get("check_inner_finger", True))
    check_outer = bool(section.get("check_outer_finger", True))
    checked = np.zeros_like(usable_bool)
    if check_inner:
        checked |= inner_sweep
    if check_outer:
        checked |= outer_sweep

    distance_map = cv2.distanceTransform(usable, cv2.DIST_L2, 5)
    distances = distance_map[checked] if np.any(checked) else np.empty(0, dtype=np.float32)
    percentile = float(np.clip(_safe_float(section.get("clearance_percentile"), 10.0), 0.0, 100.0))
    clearance_px = float(np.percentile(distances, percentile)) if distances.size else 0.0
    clearance_mm = clearance_px * max(0.0, float(mm_per_px))

    min_inner = _safe_float(section.get("minimum_inner_finger_containment"), 0.950)
    min_outer = _safe_float(section.get("minimum_outer_finger_containment"), 0.985)
    required_mm = _safe_float(section.get("minimum_wall_clearance_mm"), 8.0)
    required_px = _safe_float(section.get("minimum_wall_clearance_px"), 0.0)
    intersects = (check_inner and inner_containment < min_inner) or (check_outer and outer_containment < min_outer)
    too_close = clearance_mm < required_mm or clearance_px < required_px
    status = "intersects" if intersects else ("too_close" if too_close else "clear")
    return {
        "enabled": True,
        "status": status,
        "inner_polygon_uv": polygon.reshape(-1, 2).astype(float).tolist(),
        "inner_finger_containment": float(inner_containment),
        "outer_finger_containment": float(outer_containment),
        "clearance_percentile": float(percentile),
        "clearance_px": float(clearance_px),
        "clearance_mm": float(clearance_mm),
        "required_clearance_mm": float(required_mm),
        "required_clearance_px": float(required_px),
        "hard_reject": bool(section.get("hard_reject", True)),
    }



def _resolve_box_model_3d(config: GeometryConfig) -> Optional[BoxModel3D]:
    section = config.section("box_wall")
    if not bool(section.get("enabled", False)):
        return None
    if str(section.get("model") or "") != "calibrated_3d_cuboid":
        return None
    payload = section.get("calibrated_model")
    if not isinstance(payload, dict):
        return None
    try:
        return box_model_from_dict(payload, source_path=section.get("_resolved_calibration_file"))
    except Exception:
        return None


def _check_box_wall_3d(
    model: BoxModel3D,
    inner_pre_center: np.ndarray,
    outer_pre_center: np.ndarray,
    inner_front_center: np.ndarray,
    outer_front_center: np.ndarray,
    inner_open_insert_center: np.ndarray,
    outer_open_insert_center: np.ndarray,
    inner_closed_center: np.ndarray,
    outer_closed_center: np.ndarray,
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    finger_thickness: float,
    finger_width: float,
    config: GeometryConfig,
) -> Dict[str, Any]:
    section = config.section("box_wall")
    sample_count = _safe_int(section.get("sweep_sample_count"), 18)
    front_tolerance = _safe_float(section.get("front_entry_tolerance_mm"), 2.0)
    checks: List[Dict[str, Any]] = []
    for finger_name, centers in (
        (
            "inner",
            (
                ("approach", inner_pre_center, inner_front_center),
                ("insert", inner_front_center, inner_open_insert_center),
                ("close", inner_open_insert_center, inner_closed_center),
            ),
        ),
        (
            "outer",
            (
                ("approach", outer_pre_center, outer_front_center),
                ("insert", outer_front_center, outer_open_insert_center),
                ("close", outer_open_insert_center, outer_closed_center),
            ),
        ),
    ):
        for stage_name, start, end in centers:
            result = check_swept_prism_against_box(
                model,
                start,
                end,
                closing_axis,
                tangent_axis,
                finger_thickness,
                finger_width,
                stage=f"{finger_name}_{stage_name}",
                sample_count=sample_count,
                front_entry_tolerance_mm=front_tolerance,
            )
            result["finger"] = finger_name
            result["motion"] = stage_name
            checks.append(result)
    combined = combine_collision_checks(checks)
    combined["hard_reject"] = bool(section.get("hard_reject", True))
    combined["model_source"] = model.source_path
    combined["safety_margin_mm"] = dict(model.safety_margin_mm)
    return combined

def _plane_expected_depth_map(
    mask: np.ndarray,
    intrinsics: Mapping[str, float],
    plane: PlaneModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty(0, dtype=np.float64), xs, ys
    rays = np.column_stack(
        (
            (xs.astype(np.float64) - float(intrinsics["cx"])) / float(intrinsics["fx"]),
            (ys.astype(np.float64) - float(intrinsics["cy"])) / float(intrinsics["fy"]),
            np.ones_like(xs, dtype=np.float64),
        )
    )
    denominators = rays @ plane.normal
    expected = np.full(len(xs), np.nan, dtype=np.float64)
    good = np.abs(denominators) > 1e-8
    expected[good] = -float(plane.offset) / denominators[good]
    return expected, xs, ys


def _local_front_obstacle(
    depth: np.ndarray,
    sweep_mask: np.ndarray,
    intrinsics: Mapping[str, float],
    plane: PlaneModel,
    config: GeometryConfig,
    tilt_deg: float,
) -> Dict[str, Any]:
    depth_cfg = config.section("depth")
    observable_limit = _safe_float(depth_cfg.get("local_obstacle_observable_max_tilt_deg"), 25.0)
    if float(tilt_deg) > observable_limit:
        return {
            "status": "unknown",
            "reason": "single_view_limited_by_tilt",
            "observable_max_tilt_deg": observable_limit,
            "obstacle_ratio": None,
            "valid_pixel_count": 0,
        }
    expected, xs, ys = _plane_expected_depth_map(sweep_mask, intrinsics, plane)
    if xs.size == 0:
        return {"status": "unknown", "reason": "empty_sweep", "obstacle_ratio": None, "valid_pixel_count": 0}
    observed = depth[ys, xs].astype(np.float64)
    minimum = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    valid = (observed >= minimum) & (observed <= maximum) & np.isfinite(expected) & (expected > 0.0)
    count = int(np.count_nonzero(valid))
    if count < 8:
        return {
            "status": "unknown",
            "reason": "insufficient_depth_support",
            "obstacle_ratio": None,
            "valid_pixel_count": count,
            "total_pixel_count": int(xs.size),
        }
    margin = _safe_float(depth_cfg.get("local_obstacle_margin_mm"), 8.0)
    obstacles = observed[valid] < expected[valid] - margin
    ratio = float(np.count_nonzero(obstacles)) / float(max(1, count))
    maximum_ratio = _safe_float(depth_cfg.get("maximum_front_obstacle_ratio"), 0.18)
    return {
        "status": "clear" if ratio <= maximum_ratio else "blocked",
        "obstacle_ratio": ratio,
        "maximum_obstacle_ratio": maximum_ratio,
        "margin_mm": margin,
        "valid_pixel_count": count,
        "total_pixel_count": int(xs.size),
    }


def _rotation_matrix_and_quaternion(
    closing_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _normalize(approach_axis)
    x_raw = np.asarray(closing_axis, dtype=np.float64)
    x_axis = _normalize(x_raw - z_axis * float(np.dot(x_raw, z_axis)))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    # Re-orthogonalize x to reduce numerical drift.
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return x_axis, y_axis, z_axis, rotation, quaternion


def _robot_grasp_frame(
    candidate: Mapping[str, Any],
    config: GeometryConfig,
) -> Dict[str, Any]:
    origin = np.asarray(candidate["grasp_center_camera_mm"], dtype=np.float64)
    closing = np.asarray(candidate["closing_axis_camera"], dtype=np.float64)
    approach = np.asarray(candidate["approach_vector_camera"], dtype=np.float64)
    x_axis, y_axis, z_axis, rotation, quaternion = _rotation_matrix_and_quaternion(closing, approach)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin
    interface_cfg = config.section("robot_interface")
    return {
        "coordinate_frame": str(interface_cfg.get("camera_frame_id") or "camera_color_optical_frame"),
        "camera_optical_convention": "+X right, +Y down, +Z forward",
        "length_unit": str(interface_cfg.get("length_unit") or "mm"),
        "origin_camera_mm": _json_vector(origin),
        "x_closing_axis_camera": _json_vector(x_axis),
        "y_lateral_axis_camera": _json_vector(y_axis),
        "z_approach_axis_camera": _json_vector(z_axis),
        "rotation_matrix_rows": [[float(value) for value in row] for row in rotation.tolist()],
        "quaternion_xyzw": _json_vector(quaternion),
        "T_camera_grasp_rows": [[float(value) for value in row] for row in transform.tolist()],
        "inner_finger_side": "negative_x",
        "outer_finger_side": "positive_x",
    }


def _distance_to_mask_px(mask: np.ndarray, uv: Tuple[float, float]) -> float:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return 0.0
    if not np.any(mask):
        return float(math.hypot(mask.shape[0], mask.shape[1]))
    distance = cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[y, x])


def _distance_map_to_mask_px(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        fill = float(math.hypot(mask.shape[0], mask.shape[1]))
        return np.full(mask.shape, fill, dtype=np.float32)
    return cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)


def _distance_from_map_px(distance_map: np.ndarray, uv: Tuple[float, float]) -> float:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if not (0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]):
        return 0.0
    return float(distance_map[y, x])



def _expected_plane_depth_at_pixels(
    pixels: np.ndarray,
    intrinsics: Mapping[str, float],
    plane: PlaneModel,
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) == 0:
        return np.empty(0, dtype=np.float64)
    rays = np.column_stack(
        (
            (pixels[:, 0] - float(intrinsics["cx"])) / float(intrinsics["fx"]),
            (pixels[:, 1] - float(intrinsics["cy"])) / float(intrinsics["fy"]),
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    denominators = rays @ plane.normal
    expected = np.full(len(pixels), np.nan, dtype=np.float64)
    valid = np.abs(denominators) > 1e-8
    expected[valid] = -float(plane.offset) / denominators[valid]
    return expected


def _build_neighbor_base_cache(
    all_rings: Sequence[SegmentationInstance],
    ring_mouth_masks: Mapping[int, np.ndarray],
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    config: GeometryConfig,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Build depth point clouds once per ring before target-specific filtering.

    The old implementation reprojected every non-target ring for every matched
    target.  M36.4.1 performs the expensive mask/depth-to-3D conversion once,
    then applies the target-plane duplicate suppression lazily per selected
    target candidate.
    """
    started = time.perf_counter()
    section = config.section("neighbor_3d")
    if not bool(section.get("enabled", True)):
        return {}, {
            "enabled": False,
            "status": "disabled",
            "instance_count": len(all_rings),
            "point_count": 0,
            "build_ms": _elapsed_ms(started),
        }
    depth_cfg = config.section("depth")
    minimum_mm = _safe_float(section.get("minimum_depth_mm"), _safe_float(depth_cfg.get("minimum_mm"), 150.0))
    maximum_mm = _safe_float(section.get("maximum_depth_mm"), _safe_float(depth_cfg.get("maximum_mm"), 3000.0))
    erode_px = _safe_int(section.get("mask_erode_px"), 1)
    mouth_exclusion_px = _safe_int(section.get("neighbor_mouth_exclusion_px"), 1)
    stride = max(1, _safe_int(section.get("point_stride"), 2))
    cache: Dict[int, Dict[str, Any]] = {}
    total_points = 0
    for item in all_rings:
        mask = _erode(item.mask, erode_px)
        excluded_mouth_pixel_count = 0
        mouth_mask = ring_mouth_masks.get(int(item.instance_id))
        if isinstance(mouth_mask, np.ndarray):
            exclusion = _dilate(mouth_mask.astype(bool), mouth_exclusion_px)
            excluded_mouth_pixel_count = int(np.count_nonzero(mask & exclusion))
            mask &= ~exclusion
        points, pixels = depth_pixels_to_points(
            depth,
            mask,
            intrinsics,
            minimum_mm,
            maximum_mm,
            stride=stride,
        )
        total_points += int(len(points))
        cache[int(item.instance_id)] = {
            "instance_id": int(item.instance_id),
            "mask_area_px": int(item.area_px),
            "points_camera": points,
            "pixels_uv": pixels,
            "raw_point_count": int(len(points)),
            "excluded_neighbor_mouth_pixel_count": excluded_mouth_pixel_count,
        }
    return cache, {
        "enabled": True,
        "status": "ready",
        "instance_count": len(cache),
        "point_count": int(total_points),
        "build_ms": _elapsed_ms(started),
    }


def _prepare_neighbor_point_clouds(
    all_rings: Sequence[SegmentationInstance],
    target_ring: SegmentationInstance,
    ring_mouth_masks: Mapping[int, np.ndarray],
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    target_plane: PlaneModel,
    config: GeometryConfig,
    base_cache: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build visible per-instance point clouds for rings other than the target.

    A single aligned depth image only contains the nearest visible surface. PT
    masks can overlap in image space, so points that agree with the target
    front-pose plane are removed from an overlapping neighbor mask. This avoids
    duplicating target-surface pixels into a neighbor cloud and recreating the
    old 2-D false-positive behavior in 3-D.
    """
    section = config.section("neighbor_3d")
    enabled = bool(section.get("enabled", True))
    other_rings = [item for item in all_rings if int(item.instance_id) != int(target_ring.instance_id)]
    if not enabled:
        return [], {
            "enabled": False,
            "status": "disabled",
            "neighbor_instance_count": len(other_rings),
            "ready_instance_count": 0,
            "retained_point_count": 0,
            "instances": [],
        }

    depth_cfg = config.section("depth")
    minimum_mm = _safe_float(section.get("minimum_depth_mm"), _safe_float(depth_cfg.get("minimum_mm"), 150.0))
    maximum_mm = _safe_float(section.get("maximum_depth_mm"), _safe_float(depth_cfg.get("maximum_mm"), 3000.0))
    erode_px = _safe_int(section.get("mask_erode_px"), 1)
    mouth_exclusion_px = _safe_int(section.get("neighbor_mouth_exclusion_px"), 1)
    stride = max(1, _safe_int(section.get("point_stride"), 2))
    minimum_points = max(1, _safe_int(section.get("minimum_points_per_instance"), 12))
    maximum_points = max(minimum_points, _safe_int(section.get("maximum_points_per_instance"), 3000))
    suppress_target = bool(section.get("target_surface_exclusion_enabled", True))
    target_dilate_px = _safe_int(section.get("target_surface_exclusion_dilate_px"), 2)
    target_tolerance_mm = _safe_float(section.get("target_surface_exclusion_mm"), 12.0)
    target_region = _dilate(target_ring.mask, target_dilate_px)

    clouds: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    total_retained = 0
    for item in other_rings:
        cached = base_cache.get(int(item.instance_id)) if base_cache is not None else None
        if isinstance(cached, Mapping):
            points = np.asarray(cached.get("points_camera"), dtype=np.float64).reshape(-1, 3)
            pixels = np.asarray(cached.get("pixels_uv"), dtype=np.int64).reshape(-1, 2)
            raw_count = int(cached.get("raw_point_count", len(points)))
            excluded_mouth_pixel_count = int(cached.get("excluded_neighbor_mouth_pixel_count", 0))
        else:
            mask = _erode(item.mask, erode_px)
            excluded_mouth_pixel_count = 0
            neighbor_mouth_mask = ring_mouth_masks.get(int(item.instance_id))
            if isinstance(neighbor_mouth_mask, np.ndarray):
                exclusion = _dilate(neighbor_mouth_mask.astype(bool), mouth_exclusion_px)
                excluded_mouth_pixel_count = int(np.count_nonzero(mask & exclusion))
                mask &= ~exclusion
            points, pixels = depth_pixels_to_points(
                depth,
                mask,
                intrinsics,
                minimum_mm,
                maximum_mm,
                stride=stride,
            )
            raw_count = int(len(points))
        removed_target_surface = 0
        if suppress_target and len(points):
            xs = pixels[:, 0]
            ys = pixels[:, 1]
            in_target_projection = target_region[ys, xs]
            if np.any(in_target_projection):
                expected = _expected_plane_depth_at_pixels(pixels, intrinsics, target_plane)
                same_surface = (
                    in_target_projection
                    & np.isfinite(expected)
                    & (expected > 0.0)
                    & (np.abs(points[:, 2] - expected) <= target_tolerance_mm)
                )
                removed_target_surface = int(np.count_nonzero(same_surface))
                keep = ~same_surface
                points = points[keep]
                pixels = pixels[keep]
        if len(points) > maximum_points:
            indexes = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
            points = points[indexes]
            pixels = pixels[indexes]
        retained = int(len(points))
        total_retained += retained
        status = "ready" if retained >= minimum_points else "insufficient"
        summary = {
            "instance_id": int(item.instance_id),
            "status": status,
            "mask_area_px": int(item.area_px),
            "raw_point_count": raw_count,
            "retained_point_count": retained,
            "removed_target_surface_point_count": removed_target_surface,
            "excluded_neighbor_mouth_pixel_count": excluded_mouth_pixel_count,
            "minimum_required_points": minimum_points,
        }
        summaries.append(summary)
        clouds.append({
            **summary,
            "points_camera": points,
            "pixels_uv": pixels,
        })

    ready_count = sum(1 for row in summaries if row["status"] == "ready")
    if not other_rings:
        status = "clear_no_neighbors"
    elif ready_count:
        status = "ready"
    else:
        status = "insufficient_depth_support"
    return clouds, {
        "enabled": True,
        "status": status,
        "neighbor_instance_count": len(other_rings),
        "ready_instance_count": ready_count,
        "retained_point_count": total_retained,
        "instances": summaries,
        "target_surface_exclusion_enabled": suppress_target,
        "target_surface_exclusion_mm": target_tolerance_mm,
    }


def _orthonormal_finger_basis(
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closing = _normalize(np.asarray(closing_axis, dtype=np.float64))
    tangent_raw = np.asarray(tangent_axis, dtype=np.float64)
    tangent = _normalize(tangent_raw - closing * float(np.dot(tangent_raw, closing)))
    approach = _normalize(np.cross(closing, tangent))
    tangent = _normalize(np.cross(approach, closing))
    basis = np.column_stack((closing, tangent, approach))
    return closing, tangent, approach, basis


def _points_against_swept_finger_volume(
    points_camera: np.ndarray,
    center_points_camera: Sequence[np.ndarray],
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    finger_thickness_mm: float,
    finger_width_mm: float,
    intersection_tolerance_mm: float,
    robust_k: int,
) -> Dict[str, Any]:
    """Measure point-cloud clearance to a translated rectangular finger volume.

    The center points define the swept center region. Approach/insert stages use
    two endpoints. The close stage supplies four centers (open/closed at rim and
    inserted depth), so the complete inserted finger length is included while it
    moves laterally.
    """
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    centers = np.asarray(center_points_camera, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(centers) == 0:
        return {
            "point_count": int(len(points)),
            "inside_point_count": 0,
            "collision_point_count": 0,
            "raw_minimum_clearance_mm": None,
            "robust_minimum_clearance_mm": None,
            "nearest_point_camera_mm": None,
        }
    _, _, _, basis = _orthonormal_finger_basis(closing_axis, tangent_axis)
    reference = centers[0]
    center_local = (centers - reference) @ basis
    lower = center_local.min(axis=0)
    upper = center_local.max(axis=0)
    lower[0] -= 0.5 * float(finger_thickness_mm)
    upper[0] += 0.5 * float(finger_thickness_mm)
    lower[1] -= 0.5 * float(finger_width_mm)
    upper[1] += 0.5 * float(finger_width_mm)

    local = (points - reference) @ basis
    below = np.maximum(lower.reshape(1, 3) - local, 0.0)
    above = np.maximum(local - upper.reshape(1, 3), 0.0)
    outside = below + above
    distances = np.linalg.norm(outside, axis=1)
    inside = np.all((local >= lower.reshape(1, 3)) & (local <= upper.reshape(1, 3)), axis=1)
    tolerance = max(0.0, float(intersection_tolerance_mm))
    collision = distances <= tolerance
    sorted_distances = np.sort(distances)
    kth_index = min(max(1, int(robust_k)) - 1, len(sorted_distances) - 1)
    nearest_index = int(np.argmin(distances))
    return {
        "point_count": int(len(points)),
        "inside_point_count": int(np.count_nonzero(inside)),
        "collision_point_count": int(np.count_nonzero(collision)),
        "raw_minimum_clearance_mm": float(distances[nearest_index]),
        "robust_minimum_clearance_mm": float(sorted_distances[kth_index]),
        "nearest_point_camera_mm": points[nearest_index].astype(float).tolist(),
        "local_bounds_min_mm": lower.astype(float).tolist(),
        "local_bounds_max_mm": upper.astype(float).tolist(),
    }


def _check_neighbor_collision_3d(
    neighbor_clouds: Sequence[Mapping[str, Any]],
    stage_volumes: Sequence[Mapping[str, Any]],
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    finger_thickness_mm: float,
    finger_width_mm: float,
    config: GeometryConfig,
) -> Dict[str, Any]:
    section = config.section("neighbor_3d")
    enabled = bool(section.get("enabled", True))
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "minimum_clearance_mm": None,
            "raw_minimum_clearance_mm": None,
            "colliding_instance_ids": [],
            "checks": [],
            "hard_reject_on_intersection": False,
            "hard_reject_on_clearance": False,
            "hard_reject_on_unknown": False,
        }

    minimum_points_per_instance = max(1, _safe_int(section.get("minimum_points_per_instance"), 12))
    minimum_total_points = max(1, _safe_int(section.get("minimum_total_points"), 20))
    minimum_collision_points = max(1, _safe_int(section.get("minimum_collision_points"), 4))
    intersection_tolerance = _safe_float(section.get("intersection_tolerance_mm"), 1.5)
    minimum_clearance = _safe_float(section.get("minimum_clearance_mm"), 3.0)
    ready_clouds = [
        row for row in neighbor_clouds
        if int(row.get("retained_point_count", 0)) >= minimum_points_per_instance
        and isinstance(row.get("points_camera"), np.ndarray)
    ]
    total_points = sum(int(row.get("retained_point_count", 0)) for row in ready_clouds)
    if not neighbor_clouds:
        return {
            "enabled": True,
            "status": "clear",
            "reason": "no_other_ring_instances",
            "minimum_clearance_mm": None,
            "raw_minimum_clearance_mm": None,
            "colliding_instance_ids": [],
            "nearest_instance_id": None,
            "worst_stage": None,
            "neighbor_instance_count": 0,
            "ready_instance_count": 0,
            "retained_point_count": 0,
            "checks": [],
            "hard_reject_on_intersection": bool(section.get("hard_reject_on_intersection", True)),
            "hard_reject_on_clearance": bool(section.get("hard_reject_on_clearance", True)),
            "hard_reject_on_unknown": bool(section.get("hard_reject_on_unknown", False)),
        }
    if not ready_clouds or total_points < minimum_total_points:
        return {
            "enabled": True,
            "status": "unknown",
            "reason": "insufficient_neighbor_depth_support",
            "minimum_clearance_mm": None,
            "raw_minimum_clearance_mm": None,
            "colliding_instance_ids": [],
            "nearest_instance_id": None,
            "worst_stage": None,
            "neighbor_instance_count": len(neighbor_clouds),
            "ready_instance_count": len(ready_clouds),
            "retained_point_count": total_points,
            "checks": [],
            "hard_reject_on_intersection": bool(section.get("hard_reject_on_intersection", True)),
            "hard_reject_on_clearance": bool(section.get("hard_reject_on_clearance", True)),
            "hard_reject_on_unknown": bool(section.get("hard_reject_on_unknown", False)),
        }

    checks: List[Dict[str, Any]] = []
    colliding_ids = set()
    for stage in stage_volumes:
        centers = [np.asarray(value, dtype=np.float64) for value in stage.get("centers_camera_mm") or []]
        for cloud in ready_clouds:
            metrics = _points_against_swept_finger_volume(
                np.asarray(cloud["points_camera"], dtype=np.float64),
                centers,
                closing_axis,
                tangent_axis,
                finger_thickness_mm,
                finger_width_mm,
                intersection_tolerance,
                minimum_collision_points,
            )
            collision = int(metrics.get("collision_point_count", 0)) >= minimum_collision_points
            if collision:
                colliding_ids.add(int(cloud["instance_id"]))
            checks.append({
                "stage": str(stage.get("stage") or "unknown"),
                "finger": str(stage.get("finger") or "unknown"),
                "motion": str(stage.get("motion") or "unknown"),
                "neighbor_instance_id": int(cloud["instance_id"]),
                "neighbor_point_count": int(cloud.get("retained_point_count", 0)),
                "collision": bool(collision),
                **metrics,
            })

    finite_robust = [
        row for row in checks if row.get("robust_minimum_clearance_mm") is not None
    ]
    nearest = min(
        finite_robust,
        key=lambda row: float(row["robust_minimum_clearance_mm"]),
    ) if finite_robust else None
    finite_raw = [
        float(row["raw_minimum_clearance_mm"])
        for row in checks if row.get("raw_minimum_clearance_mm") is not None
    ]
    robust_minimum = float(nearest["robust_minimum_clearance_mm"]) if nearest else None
    if colliding_ids:
        status = "intersects"
    elif robust_minimum is not None and robust_minimum < minimum_clearance:
        status = "too_close"
    else:
        status = "clear"
    stage_summaries: List[Dict[str, Any]] = []
    for stage_name in sorted({str(row.get("stage")) for row in checks}):
        rows = [row for row in checks if str(row.get("stage")) == stage_name]
        stage_nearest = min(
            (row for row in rows if row.get("robust_minimum_clearance_mm") is not None),
            key=lambda row: float(row["robust_minimum_clearance_mm"]),
            default=None,
        )
        stage_colliding = sorted({
            int(row["neighbor_instance_id"]) for row in rows if bool(row.get("collision"))
        })
        stage_summaries.append({
            "stage": stage_name,
            "finger": stage_nearest.get("finger") if stage_nearest else None,
            "motion": stage_nearest.get("motion") if stage_nearest else None,
            "status": "intersects" if stage_colliding else (
                "too_close"
                if stage_nearest is not None
                and float(stage_nearest["robust_minimum_clearance_mm"]) < minimum_clearance
                else "clear"
            ),
            "minimum_clearance_mm": (
                float(stage_nearest["robust_minimum_clearance_mm"]) if stage_nearest else None
            ),
            "raw_minimum_clearance_mm": min(
                (float(row["raw_minimum_clearance_mm"]) for row in rows if row.get("raw_minimum_clearance_mm") is not None),
                default=None,
            ),
            "nearest_instance_id": (
                int(stage_nearest["neighbor_instance_id"]) if stage_nearest else None
            ),
            "colliding_instance_ids": stage_colliding,
            "collision_point_count": sum(int(row.get("collision_point_count", 0)) for row in rows),
        })

    include_full_checks = bool(section.get("include_instance_checks_in_json", False))
    result = {
        "enabled": True,
        "status": status,
        "minimum_clearance_mm": robust_minimum,
        "raw_minimum_clearance_mm": min(finite_raw) if finite_raw else None,
        "required_clearance_mm": float(minimum_clearance),
        "intersection_tolerance_mm": float(intersection_tolerance),
        "minimum_collision_points": int(minimum_collision_points),
        "colliding_instance_ids": sorted(colliding_ids),
        "nearest_instance_id": int(nearest["neighbor_instance_id"]) if nearest else None,
        "worst_stage": nearest.get("stage") if nearest else None,
        "neighbor_instance_count": len(neighbor_clouds),
        "ready_instance_count": len(ready_clouds),
        "retained_point_count": total_points,
        "checks": checks if include_full_checks else stage_summaries,
        "checks_are_stage_summaries": not include_full_checks,
        "hard_reject_on_intersection": bool(section.get("hard_reject_on_intersection", True)),
        "hard_reject_on_clearance": bool(section.get("hard_reject_on_clearance", True)),
        "hard_reject_on_unknown": bool(section.get("hard_reject_on_unknown", False)),
    }
    if not include_full_checks:
        result["_debug"] = {"instance_checks": checks}
    return result


def _clock_candidate(
    clock: Mapping[str, Any],
    ring: SegmentationInstance,
    mouth: SegmentationInstance,
    other_ring_mask: np.ndarray,
    neighbor_clouds: Sequence[Mapping[str, Any]],
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    pose_plane: PlaneModel,
    center_camera: np.ndarray,
    tilt_deg: float,
    config: GeometryConfig,
    *,
    evaluation_level: str = "full",
    other_ring_distance_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    candidate_started = time.perf_counter()
    timing_ms: Dict[str, float] = {}
    light_only = str(evaluation_level).strip().lower() != "full"
    boundary_started = time.perf_counter()
    gripper = config.section("gripper")
    candidate_cfg = config.section("candidate")
    object_cfg = config.section("object_geometry")
    reasons: List[str] = []
    warnings: List[str] = []
    angle_deg = float(clock["image_angle_deg_from_positive_x"])
    center_uv = mask_centroid(mouth.mask)

    def fail_early(reason: str) -> Dict[str, Any]:
        timing_ms["boundary_and_2d_sweep_ms"] = _elapsed_ms(boundary_started)
        timing_ms["total_ms"] = _elapsed_ms(candidate_started)
        return {
            **dict(clock),
            "evaluation_stage": "light" if light_only else "full",
            "full_evaluated": not light_only,
            "light_valid": False,
            "valid": False,
            "score": 0.0,
            "warnings": list(warnings),
            "rejection_reasons": [reason],
            "timing_ms": timing_ms,
        }

    inner_info = _mouth_boundary_on_ray(mouth.mask, center_uv, angle_deg)
    if inner_info is None:
        return fail_early("mouth_boundary_unavailable")
    inner_uv = inner_info["uv"]
    inner_camera = ray_plane_intersection(inner_uv, intrinsics, pose_plane)
    if inner_camera is None:
        return fail_early("inner_boundary_ray_plane_failed")

    pixel_radius = max(1e-6, float(inner_info["distance_px"]))
    radial_mm = float(np.linalg.norm(inner_camera - center_camera))
    mm_per_px = radial_mm / pixel_radius
    maximum_wall = _safe_float(object_cfg.get("maximum_wall_thickness_mm"), 45.0)
    search_ratio = _safe_float(object_cfg.get("maximum_outer_search_radius_ratio"), 1.35)
    maximum_search_px = min(
        maximum_wall / max(mm_per_px, 1e-6) * 1.10,
        pixel_radius * max(0.50, search_ratio),
    )
    maximum_gap_px = _safe_int(object_cfg.get("maximum_ring_mask_gap_px"), 2)
    outer_boundary_source = "raw_ring_mask"
    outer_info = _outer_boundary_on_ray(
        ring.mask,
        center_uv,
        angle_deg,
        float(inner_info["distance_px"]),
        maximum_search_px,
        maximum_gap_px,
    )
    if outer_info is None:
        association_cfg = config.section("association")
        envelope = filled_outer_envelope(
            ring.mask,
            dilate_px=0,
            close_px=_safe_int(association_cfg.get("envelope_close_px"), 3),
            minimum_component_area_ratio=_safe_float(association_cfg.get("minimum_component_area_ratio"), 0.03),
        )
        outer_info = _outer_boundary_on_ray(
            envelope,
            center_uv,
            angle_deg,
            float(inner_info["distance_px"]),
            maximum_search_px,
            maximum_gap_px,
        )
        if outer_info is not None:
            outer_boundary_source = "filled_envelope_fallback"
            warnings.append("outer_boundary_used_filled_envelope")
    if outer_info is None:
        return fail_early("outer_rim_boundary_unavailable")
    if bool(outer_info.get("ambiguous")):
        reasons.append("outer_rim_boundary_ambiguous")
    outer_uv = outer_info["uv"]
    outer_camera = ray_plane_intersection(outer_uv, intrinsics, pose_plane)
    if outer_camera is None:
        return fail_early("outer_boundary_ray_plane_failed")

    radial_vector = outer_camera - inner_camera
    wall_thickness = float(np.linalg.norm(radial_vector))
    if wall_thickness <= 1e-6:
        return fail_early("invalid_wall_thickness")
    closing_axis = radial_vector / wall_thickness  # inner -> outer
    normal_toward_camera = _normalize(pose_plane.normal)
    approach = -normal_toward_camera
    tangent = _normalize(np.cross(approach, closing_axis))
    closing_axis = _normalize(np.cross(tangent, approach))

    min_wall = _safe_float(object_cfg.get("minimum_wall_thickness_mm"), 8.0)
    max_wall = _safe_float(object_cfg.get("maximum_wall_thickness_mm"), 45.0)
    if not (min_wall <= wall_thickness <= max_wall):
        reasons.append("wall_thickness_out_of_range")

    minimum_opening = _safe_float(gripper.get("minimum_opening_mm"), 10.0)
    maximum_opening = _safe_float(gripper.get("maximum_opening_mm"), 80.0)
    desired_compression = _safe_float(gripper.get("wall_compression_mm"), 1.5)
    desired_target_gap = wall_thickness - 2.0 * desired_compression

    closing_margin = _safe_float(gripper.get("closing_limit_margin_mm"), 0.5)
    closing_min = minimum_opening + closing_margin
    closing_max = maximum_opening - closing_margin
    if closing_min > closing_max:
        reasons.append("invalid_closing_opening_limits")
        target_gap = desired_target_gap
    else:
        target_gap = float(np.clip(desired_target_gap, closing_min, closing_max))
    actual_compression = 0.5 * (wall_thickness - target_gap)
    min_compression = _safe_float(gripper.get("minimum_contact_compression_mm"), 0.5)
    max_compression = _safe_float(gripper.get("maximum_contact_compression_mm"), 3.0)
    if actual_compression < min_compression or actual_compression > max_compression:
        reasons.append("contact_compression_out_of_range")

    preopen_clearance = _safe_float(gripper.get("preopen_clearance_mm"), 6.0)
    approach_opening = wall_thickness + 2.0 * preopen_clearance
    approach_margin = _safe_float(gripper.get("approach_limit_margin_mm"), 3.0)
    approach_max = maximum_opening - approach_margin
    if approach_opening > approach_max:
        reasons.append("approach_opening_exceeds_limit")

    rim_insert = _safe_float(gripper.get("rim_insert_depth_mm"), 20.0)
    front_midpoint = 0.5 * (inner_camera + outer_camera)
    grasp_center = front_midpoint + approach * rim_insert
    pregrasp = front_midpoint - approach * _safe_float(gripper.get("pregrasp_offset_mm"), 90.0)
    inner_contact = inner_camera + closing_axis * actual_compression + approach * rim_insert
    outer_contact = outer_camera - closing_axis * actual_compression + approach * rim_insert

    finger_thickness = _safe_float(gripper.get("finger_thickness_mm"), 17.0)
    finger_width = _safe_float(gripper.get("finger_width_mm"), 20.0)
    inner_front_center = front_midpoint - closing_axis * (approach_opening * 0.5 + finger_thickness * 0.5)
    outer_front_center = front_midpoint + closing_axis * (approach_opening * 0.5 + finger_thickness * 0.5)
    inner_final_center = inner_front_center + approach * rim_insert
    outer_final_center = outer_front_center + approach * rim_insert
    pregrasp_midpoint = front_midpoint - approach * _safe_float(gripper.get("pregrasp_offset_mm"), 90.0)
    inner_pre_center = pregrasp_midpoint - closing_axis * (approach_opening * 0.5 + finger_thickness * 0.5)
    outer_pre_center = pregrasp_midpoint + closing_axis * (approach_opening * 0.5 + finger_thickness * 0.5)
    inner_closed_front_center = front_midpoint - closing_axis * (target_gap * 0.5 + finger_thickness * 0.5)
    outer_closed_front_center = front_midpoint + closing_axis * (target_gap * 0.5 + finger_thickness * 0.5)
    inner_closed_center = grasp_center - closing_axis * (target_gap * 0.5 + finger_thickness * 0.5)
    outer_closed_center = grasp_center + closing_axis * (target_gap * 0.5 + finger_thickness * 0.5)
    inner_polygon = _finger_sweep_polygon(
        inner_front_center,
        inner_final_center,
        closing_axis,
        tangent,
        finger_thickness,
        finger_width,
        intrinsics,
    )
    outer_polygon = _finger_sweep_polygon(
        outer_front_center,
        outer_final_center,
        closing_axis,
        tangent,
        finger_thickness,
        finger_width,
        intrinsics,
    )
    inner_sweep = _polygon_mask(mouth.mask.shape, inner_polygon)
    outer_sweep = _polygon_mask(mouth.mask.shape, outer_polygon)
    inner_area = max(1, int(np.count_nonzero(inner_sweep)))
    outer_area = max(1, int(np.count_nonzero(outer_sweep)))
    inner_containment = float(np.count_nonzero(inner_sweep & mouth.mask)) / float(inner_area)
    other_inner_overlap = float(np.count_nonzero(inner_sweep & other_ring_mask)) / float(inner_area)
    other_outer_overlap = float(np.count_nonzero(outer_sweep & other_ring_mask)) / float(outer_area)
    maximum_overlap = _safe_float(candidate_cfg.get("maximum_other_ring_overlap_ratio"), 0.10)
    minimum_containment = _safe_float(candidate_cfg.get("minimum_inner_finger_mouth_containment"), 0.65)
    if inner_containment < minimum_containment:
        reasons.append("inner_finger_does_not_fit_mouth")
    neighbor_2d_overlap = max(other_inner_overlap, other_outer_overlap)
    overlap_mode = str(candidate_cfg.get("neighbor_2d_overlap_mode") or "warning_only")
    if neighbor_2d_overlap > maximum_overlap:
        if overlap_mode == "hard_reject":
            reasons.append("neighbor_ring_overlap_too_large")
        elif overlap_mode == "warning_only":
            warnings.append("neighbor_2d_overlap_warning")

    timing_ms["boundary_and_2d_sweep_ms"] = _elapsed_ms(boundary_started)

    # M36.4.1 lightweight ranking stage.  It deliberately avoids calibrated
    # box sweeps, neighbor-cloud volume checks and the complete gripper model.
    # Those checks are performed only for the globally best-ranked candidates.
    if light_only:
        local_started = time.perf_counter()
        border_margin = min(
            _polygon_border_margin_px(inner_polygon, mouth.mask.shape),
            _polygon_border_margin_px(outer_polygon, mouth.mask.shape),
        )
        minimum_border = _safe_float(candidate_cfg.get("minimum_image_border_margin_px"), 3.0)
        if border_margin < minimum_border:
            reasons.append("finger_sweep_outside_image")
        distance_map = (
            other_ring_distance_map
            if isinstance(other_ring_distance_map, np.ndarray)
            else _distance_map_to_mask_px(other_ring_mask)
        )
        outer_neighbor_px = _distance_from_map_px(distance_map, outer_uv)
        inner_neighbor_px = _distance_from_map_px(distance_map, inner_uv)
        neighbor_2d_clearance_mm = min(outer_neighbor_px, inner_neighbor_px) * mm_per_px
        minimum_neighbor = _safe_float(candidate_cfg.get("minimum_neighbor_clearance_mm"), 2.0)
        clearance_mode = str(candidate_cfg.get("neighbor_2d_clearance_mode") or "warning_only")
        if neighbor_2d_clearance_mm < minimum_neighbor:
            if clearance_mode == "hard_reject":
                reasons.append("neighbor_clearance_too_small")
            elif clearance_mode == "warning_only":
                warnings.append("neighbor_2d_clearance_warning")
        combined_sweep = inner_sweep | outer_sweep
        front_obstacle = _local_front_obstacle(
            depth,
            combined_sweep,
            intrinsics,
            pose_plane,
            config,
            tilt_deg,
        )
        if front_obstacle.get("status") == "blocked":
            if bool(candidate_cfg.get("hard_reject_front_obstacle", False)):
                reasons.append("local_front_obstacle")
            else:
                warnings.append("local_front_obstacle_unverified_stage1")
        elif front_obstacle.get("status") == "unknown":
            warnings.append("local_depth_clearance_unknown")
        safe_tilt = _safe_float(gripper.get("robot_safe_max_tilt_deg"), 30.0)
        if tilt_deg > safe_tilt:
            warnings.append("tilt_above_initial_robot_safe_limit")
        timing_ms["local_depth_and_clearance_ms"] = _elapsed_ms(local_started)

        opening_margin = min(target_gap - minimum_opening, maximum_opening - target_gap)
        opening_score = float(np.clip(opening_margin / max(1.0, 0.5 * (maximum_opening - minimum_opening)), 0.0, 1.0))
        obstacle_status = str(front_obstacle.get("status") or "unknown")
        obstacle_score = 1.0 if obstacle_status == "clear" else (0.45 if obstacle_status == "unknown" else 0.0)
        tilt_limit = _safe_float(gripper.get("geometry_candidate_max_tilt_deg"), 45.0)
        tilt_score = float(np.clip(1.0 - tilt_deg / max(1.0, tilt_limit), 0.0, 1.0))
        confidence_score = float(np.clip(0.5 * (ring.confidence + mouth.confidence), 0.0, 1.0))
        border_score = float(np.clip(border_margin / 30.0, 0.0, 1.0))
        neighbor_score = float(np.clip(neighbor_2d_clearance_mm / 30.0, 0.0, 1.0))
        light_weights = {
            "inner_containment": 0.28,
            "neighbor_clearance": 0.20,
            "opening_margin": 0.18,
            "local_depth_clearance": 0.12,
            "lower_tilt": 0.10,
            "segmentation_confidence": 0.07,
            "image_border": 0.05,
        }
        configured_light = candidate_cfg.get("light_score_weights")
        if isinstance(configured_light, Mapping):
            light_weights.update({str(k): _safe_float(v, light_weights.get(str(k), 0.0)) for k, v in configured_light.items()})
        weight_sum = max(1e-6, sum(max(0.0, float(value)) for value in light_weights.values()))
        light_score = 100.0 * (
            light_weights["inner_containment"] * float(np.clip(inner_containment, 0.0, 1.0))
            + light_weights["neighbor_clearance"] * neighbor_score
            + light_weights["opening_margin"] * opening_score
            + light_weights["local_depth_clearance"] * obstacle_score
            + light_weights["lower_tilt"] * tilt_score
            + light_weights["segmentation_confidence"] * confidence_score
            + light_weights["image_border"] * border_score
        ) / weight_sum
        light_valid = len(reasons) == 0
        if not light_valid:
            light_score *= 0.25
        deferred = {
            "enabled": True,
            "status": "deferred_by_staged_evaluation",
        }
        result: Dict[str, Any] = {
            **dict(clock),
            "evaluation_stage": "light",
            "full_evaluated": False,
            "light_valid": bool(light_valid),
            "valid": False,
            "score": float(light_score),
            "light_score": float(light_score),
            "warnings": warnings,
            "rejection_reasons": reasons,
            "inner_boundary_uv": _json_uv(inner_uv),
            "outer_boundary_uv": _json_uv(outer_uv),
            "inner_boundary_camera_mm": _json_vector(inner_camera),
            "outer_boundary_camera_mm": _json_vector(outer_camera),
            "rim_plane_midpoint_camera_mm": _json_vector(front_midpoint),
            "grasp_center_camera_mm": _json_vector(grasp_center),
            "pregrasp_center_camera_mm": _json_vector(pregrasp),
            "inner_contact_camera_mm": _json_vector(inner_contact),
            "outer_contact_camera_mm": _json_vector(outer_contact),
            "approach_vector_camera": _json_vector(approach),
            "closing_axis_camera": _json_vector(closing_axis),
            "lateral_axis_camera": _json_vector(tangent),
            "wall_thickness_mm": float(wall_thickness),
            "desired_target_closing_gap_mm": float(desired_target_gap),
            "target_closing_gap_mm": float(target_gap),
            "desired_wall_compression_each_side_mm": float(desired_compression),
            "actual_wall_compression_each_side_mm": float(actual_compression),
            "approach_opening_mm": float(approach_opening),
            "opening_margin_mm": float(opening_margin),
            "rim_insert_depth_mm": float(rim_insert),
            "inner_finger_mouth_containment": float(inner_containment),
            "other_ring_overlap_ratio": float(neighbor_2d_overlap),
            "neighbor_2d_overlap_mode": overlap_mode,
            "neighbor_2d_clearance_mm": float(neighbor_2d_clearance_mm),
            "neighbor_2d_clearance_mode": clearance_mode,
            "neighbor_3d": dict(deferred),
            "neighbor_3d_status": deferred["status"],
            "full_gripper_static": dict(deferred),
            "full_gripper_static_status": deferred["status"],
            "full_gripper_motion": dict(deferred),
            "full_gripper_motion_status": deferred["status"],
            "neighbor_clearance_mm": float(neighbor_2d_clearance_mm),
            "image_border_margin_px": float(border_margin),
            "box_wall": dict(deferred),
            "box_wall_clearance_mm": None,
            "box_wall_status": deferred["status"],
            "local_front_obstacle": front_obstacle,
            "inner_finger_sweep_polygon_uv": inner_polygon.reshape(-1, 2).astype(float).tolist() if inner_polygon is not None else None,
            "outer_finger_sweep_polygon_uv": outer_polygon.reshape(-1, 2).astype(float).tolist() if outer_polygon is not None else None,
            "outer_boundary_ambiguous": bool(outer_info.get("ambiguous")),
            "outer_boundary_source": outer_boundary_source,
        }
        result["grasp_frame_camera"] = _robot_grasp_frame(result, config)
        timing_ms["total_ms"] = _elapsed_ms(candidate_started)
        result["timing_ms"] = timing_ms
        return result

    box_started = time.perf_counter()
    box_section = config.section("box_wall")
    box_model_name = str(box_section.get("model") or "disabled")
    calibrated_box: Optional[BoxModel3D] = None
    if box_model_name == "calibrated_3d_cuboid" and bool(box_section.get("enabled", False)):
        calibrated_box = _resolve_box_model_3d(config)
        if calibrated_box is None:
            box_wall = {
                "enabled": True,
                "model_type": "calibrated_3d_cuboid",
                "status": "unconfigured",
                "hard_reject": bool(box_section.get("hard_reject", True)),
                "minimum_clearance_mm": None,
                "clearance_mm": None,
            }
        else:
            capture_validation = validate_model_for_capture(
                calibrated_box,
                depth.shape,
                intrinsics,
                resolution_tolerance_px=_safe_int(box_section.get("resolution_tolerance_px"), 0),
                intrinsics_relative_tolerance=_safe_float(box_section.get("intrinsics_relative_tolerance"), 0.02),
            )
            if not bool(capture_validation.get("valid")):
                box_wall = {
                    "enabled": True,
                    "model_type": "calibrated_3d_cuboid",
                    "status": "capture_mismatch",
                    "hard_reject": bool(box_section.get("hard_reject", True)),
                    "minimum_clearance_mm": None,
                    "clearance_mm": None,
                    "capture_validation": capture_validation,
                }
            else:
                box_wall = _check_box_wall_3d(
                    calibrated_box,
                    inner_pre_center,
                    outer_pre_center,
                    inner_front_center,
                    outer_front_center,
                    inner_final_center,
                    outer_final_center,
                    inner_closed_center,
                    outer_closed_center,
                    closing_axis,
                    tangent,
                    finger_thickness,
                    finger_width,
                    config,
                )
                box_wall["capture_validation"] = capture_validation
                box_wall["clearance_mm"] = box_wall.get("minimum_clearance_mm")
                box_wall["clearance_px"] = None
    else:
        box_wall = _check_box_wall_clearance(inner_sweep, outer_sweep, mm_per_px, config)
    timing_ms["box_wall_ms"] = _elapsed_ms(box_started)
    box_status = str(box_wall.get("status") or "disabled")
    is_3d_box = box_model_name == "calibrated_3d_cuboid"
    if box_status in {"unconfigured", "capture_mismatch"}:
        if is_3d_box:
            reason = "box_3d_capture_mismatch" if box_status == "capture_mismatch" else "box_3d_model_unconfigured"
        else:
            reason = "box_wall_model_unconfigured"
        if bool(box_wall.get("hard_reject")):
            reasons.append(reason)
        else:
            warnings.append(reason)
    elif box_status == "intersects":
        reason = "finger_sweep_intersects_3d_box_wall" if is_3d_box else "finger_sweep_intersects_box_wall"
        if bool(box_wall.get("hard_reject")):
            reasons.append(reason)
        else:
            warnings.append(reason)
    elif box_status == "too_close":
        reason = "box_3d_clearance_too_small" if is_3d_box else "box_wall_clearance_too_small"
        if bool(box_wall.get("hard_reject")):
            reasons.append(reason)
        else:
            warnings.append(reason)

    stage_volumes = [
        {
            "stage": "inner_approach",
            "finger": "inner",
            "motion": "approach",
            "centers_camera_mm": [inner_pre_center, inner_front_center],
        },
        {
            "stage": "inner_insert",
            "finger": "inner",
            "motion": "insert",
            "centers_camera_mm": [inner_front_center, inner_final_center],
        },
        {
            "stage": "inner_close",
            "finger": "inner",
            "motion": "close",
            "centers_camera_mm": [
                inner_front_center,
                inner_final_center,
                inner_closed_front_center,
                inner_closed_center,
            ],
        },
        {
            "stage": "outer_approach",
            "finger": "outer",
            "motion": "approach",
            "centers_camera_mm": [outer_pre_center, outer_front_center],
        },
        {
            "stage": "outer_insert",
            "finger": "outer",
            "motion": "insert",
            "centers_camera_mm": [outer_front_center, outer_final_center],
        },
        {
            "stage": "outer_close",
            "finger": "outer",
            "motion": "close",
            "centers_camera_mm": [
                outer_front_center,
                outer_final_center,
                outer_closed_front_center,
                outer_closed_center,
            ],
        },
    ]
    neighbor_started = time.perf_counter()
    neighbor_3d = _check_neighbor_collision_3d(
        neighbor_clouds,
        stage_volumes,
        closing_axis,
        tangent,
        finger_thickness,
        finger_width,
        config,
    )
    timing_ms["neighbor_3d_ms"] = _elapsed_ms(neighbor_started)
    neighbor_3d_status = str(neighbor_3d.get("status") or "disabled")
    if neighbor_3d_status == "intersects":
        if bool(neighbor_3d.get("hard_reject_on_intersection", True)):
            reasons.append("neighbor_3d_finger_collision")
        else:
            warnings.append("neighbor_3d_finger_collision")
    elif neighbor_3d_status == "too_close":
        if bool(neighbor_3d.get("hard_reject_on_clearance", True)):
            reasons.append("neighbor_3d_clearance_too_small")
        else:
            warnings.append("neighbor_3d_clearance_too_small")
    elif neighbor_3d_status == "unknown":
        if bool(neighbor_3d.get("hard_reject_on_unknown", False)):
            reasons.append("neighbor_3d_unknown")
        else:
            warnings.append("neighbor_3d_unknown")

    static_started = time.perf_counter()
    full_static = check_full_gripper_static_final_pose(
        grasp_center,
        closing_axis,
        tangent,
        approach,
        target_gap,
        calibrated_box,
        neighbor_clouds,
        config.section("gripper_geometry_3d"),
        config.section("full_gripper_static_collision"),
        intrinsics=intrinsics,
    )
    timing_ms["full_gripper_static_ms"] = _elapsed_ms(static_started)
    static_status = str(full_static.get("status") or "disabled")
    static_box_status = str(full_static.get("box_status") or "disabled")
    static_neighbor_status = str(full_static.get("neighbor_status") or "disabled")
    if static_status == "unconfigured":
        if bool(full_static.get("hard_reject_on_unconfigured", True)):
            reasons.append("full_gripper_static_model_unconfigured")
        else:
            warnings.append("full_gripper_static_model_unconfigured")
    if bool(full_static.get("hard_reject_box")):
        if static_box_status == "intersects":
            reasons.append("full_gripper_static_box_collision")
        elif static_box_status == "too_close":
            reasons.append("full_gripper_static_box_clearance_too_small")
        elif static_box_status == "unconfigured":
            reasons.append("full_gripper_static_box_unconfigured")
    elif static_box_status not in {"clear", "disabled", "outside_front"}:
        warnings.append("full_gripper_static_box_" + static_box_status)
    if bool(full_static.get("hard_reject_neighbor")):
        if static_neighbor_status == "intersects":
            reasons.append("full_gripper_static_neighbor_collision")
        elif static_neighbor_status == "too_close":
            reasons.append("full_gripper_static_neighbor_clearance_too_small")
        elif static_neighbor_status == "unknown":
            reasons.append("full_gripper_static_neighbor_unknown")
    elif static_neighbor_status not in {"clear", "disabled"}:
        warnings.append("full_gripper_static_neighbor_" + static_neighbor_status)

    border_margin = min(
        _polygon_border_margin_px(inner_polygon, mouth.mask.shape),
        _polygon_border_margin_px(outer_polygon, mouth.mask.shape),
    )
    minimum_border = _safe_float(candidate_cfg.get("minimum_image_border_margin_px"), 3.0)
    if border_margin < minimum_border:
        reasons.append("finger_sweep_outside_image")

    local_started = time.perf_counter()
    distance_map = (
        other_ring_distance_map
        if isinstance(other_ring_distance_map, np.ndarray)
        else _distance_map_to_mask_px(other_ring_mask)
    )
    outer_neighbor_px = _distance_from_map_px(distance_map, outer_uv)
    inner_neighbor_px = _distance_from_map_px(distance_map, inner_uv)
    neighbor_2d_clearance_mm = min(outer_neighbor_px, inner_neighbor_px) * mm_per_px
    minimum_neighbor = _safe_float(candidate_cfg.get("minimum_neighbor_clearance_mm"), 2.0)
    clearance_mode = str(candidate_cfg.get("neighbor_2d_clearance_mode") or "warning_only")
    if neighbor_2d_clearance_mm < minimum_neighbor:
        if clearance_mode == "hard_reject":
            reasons.append("neighbor_clearance_too_small")
        elif clearance_mode == "warning_only":
            warnings.append("neighbor_2d_clearance_warning")

    combined_sweep = inner_sweep | outer_sweep
    front_obstacle = _local_front_obstacle(
        depth,
        combined_sweep,
        intrinsics,
        pose_plane,
        config,
        tilt_deg,
    )
    if front_obstacle.get("status") == "blocked":
        if bool(candidate_cfg.get("hard_reject_front_obstacle", False)):
            reasons.append("local_front_obstacle")
        else:
            warnings.append("local_front_obstacle_unverified_stage1")
    elif front_obstacle.get("status") == "unknown":
        warnings.append("local_depth_clearance_unknown")

    safe_tilt = _safe_float(gripper.get("robot_safe_max_tilt_deg"), 30.0)
    if tilt_deg > safe_tilt:
        warnings.append("tilt_above_initial_robot_safe_limit")
    timing_ms["local_depth_and_clearance_ms"] = _elapsed_ms(local_started)

    motion_started = time.perf_counter()
    motion_cfg = config.section("full_gripper_motion_collision")
    if reasons and bool(motion_cfg.get("skip_if_prerequisite_failed", True)):
        full_motion = {
            "enabled": bool(motion_cfg.get("enabled", True)),
            "status": "skipped_prerequisite_failed",
            "motion_scope": "pregrasp_to_grasp_only",
            "pregrasp_path_checked": False,
            "post_grasp_lift_checked": False,
            "prerequisite_rejection_reasons": list(reasons),
        }
    else:
        full_motion = check_full_gripper_pregrasp_motion(
            grasp_center,
            closing_axis,
            tangent,
            approach,
            target_gap,
            approach_opening,
            _safe_float(gripper.get("pregrasp_offset_mm"), 90.0),
            _safe_float(motion_cfg.get("open_start_offset_mm"), 25.0),
            rim_insert,
            calibrated_box,
            neighbor_clouds,
            config.section("gripper_geometry_3d"),
            config.section("full_gripper_static_collision"),
            motion_cfg,
            intrinsics=intrinsics,
        )
    motion_status = str(full_motion.get("status") or "disabled")
    motion_box_status = str(full_motion.get("box_status") or "disabled")
    motion_neighbor_status = str(full_motion.get("neighbor_status") or "disabled")
    if motion_status == "unconfigured":
        if bool(full_motion.get("hard_reject_on_unconfigured", True)):
            reasons.append("full_gripper_motion_model_unconfigured")
        else:
            warnings.append("full_gripper_motion_model_unconfigured")
    if bool(full_motion.get("hard_reject_box")):
        if motion_box_status == "intersects":
            reasons.append("full_gripper_motion_box_collision")
        elif motion_box_status == "too_close":
            reasons.append("full_gripper_motion_box_clearance_too_small")
        elif motion_box_status == "unconfigured":
            reasons.append("full_gripper_motion_box_unconfigured")
    elif motion_box_status not in {"clear", "disabled", "outside_front"} and motion_status != "skipped_prerequisite_failed":
        warnings.append("full_gripper_motion_box_" + motion_box_status)
    if bool(full_motion.get("hard_reject_neighbor")):
        if motion_neighbor_status == "intersects":
            reasons.append("full_gripper_motion_neighbor_collision")
        elif motion_neighbor_status == "too_close":
            reasons.append("full_gripper_motion_neighbor_clearance_too_small")
        elif motion_neighbor_status == "unknown":
            reasons.append("full_gripper_motion_neighbor_unknown")
    elif motion_neighbor_status not in {"clear", "disabled"} and motion_status != "skipped_prerequisite_failed":
        warnings.append("full_gripper_motion_neighbor_" + motion_neighbor_status)
    timing_ms["full_gripper_motion_ms"] = _elapsed_ms(motion_started)

    opening_margin = min(target_gap - minimum_opening, maximum_opening - target_gap)
    opening_score = float(np.clip(opening_margin / max(1.0, 0.5 * (maximum_opening - minimum_opening)), 0.0, 1.0))
    neighbor_3d_clearance = neighbor_3d.get("minimum_clearance_mm")
    neighbor_3d_cfg = config.section("neighbor_3d")
    neighbor_3d_saturation = max(1.0, _safe_float(neighbor_3d_cfg.get("score_saturation_mm"), 30.0))
    if neighbor_3d_status == "clear":
        if neighbor_3d_clearance is None:
            neighbor_score = 1.0
        else:
            neighbor_score = float(np.clip(float(neighbor_3d_clearance) / neighbor_3d_saturation, 0.0, 1.0))
    elif neighbor_3d_status == "unknown":
        neighbor_score = _safe_float(neighbor_3d_cfg.get("unknown_score"), 0.35)
    else:
        neighbor_score = 0.0
    obstacle_status = str(front_obstacle.get("status") or "unknown")
    obstacle_score = 1.0 if obstacle_status == "clear" else (0.45 if obstacle_status == "unknown" else 0.0)
    tilt_limit = _safe_float(gripper.get("geometry_candidate_max_tilt_deg"), 45.0)
    tilt_score = float(np.clip(1.0 - tilt_deg / max(1.0, tilt_limit), 0.0, 1.0))
    confidence_score = float(np.clip(0.5 * (ring.confidence + mouth.confidence), 0.0, 1.0))
    border_score = float(np.clip(border_margin / 30.0, 0.0, 1.0))
    wall_cfg = config.section("box_wall")
    wall_clearance = box_wall.get("clearance_mm")
    if wall_clearance is None:
        wall_clearance = box_wall.get("minimum_clearance_mm")
    wall_saturation = max(1.0, _safe_float(wall_cfg.get("score_saturation_mm"), 35.0))
    wall_saturation_px = max(1.0, _safe_float(wall_cfg.get("score_saturation_px"), 24.0))
    wall_clearance_px = box_wall.get("clearance_px")
    if not bool(box_wall.get("enabled")):
        wall_score = 1.0
    elif str(box_wall.get("model_type") or "") == "calibrated_3d_cuboid":
        wall_score = float(np.clip(float(wall_clearance or 0.0) / wall_saturation, 0.0, 1.0))
    else:
        wall_score_mm = float(np.clip(float(wall_clearance or 0.0) / wall_saturation, 0.0, 1.0))
        wall_score_px = float(np.clip(float(wall_clearance_px or 0.0) / wall_saturation_px, 0.0, 1.0))
        wall_score = min(wall_score_mm, wall_score_px)
    if box_status in {"intersects", "unconfigured", "capture_mismatch"}:
        wall_score = 0.0
    weights = dict(candidate_cfg.get("score_weights") or {})
    score = 100.0 * (
        _safe_float(weights.get("box_wall_clearance"), 0.22) * wall_score
        + _safe_float(weights.get("inner_containment"), 0.18) * float(np.clip(inner_containment, 0.0, 1.0))
        + _safe_float(weights.get("neighbor_clearance"), 0.16) * neighbor_score
        + _safe_float(weights.get("opening_margin"), 0.14) * opening_score
        + _safe_float(weights.get("local_depth_clearance"), 0.10) * obstacle_score
        + _safe_float(weights.get("lower_tilt"), 0.10) * tilt_score
        + _safe_float(weights.get("segmentation_confidence"), 0.06) * confidence_score
        + _safe_float(weights.get("image_border"), 0.04) * border_score
    )
    static_cfg = config.section("full_gripper_static_collision")
    static_weight = float(np.clip(_safe_float(static_cfg.get("score_weight"), 0.20), 0.0, 1.0))
    static_saturation = max(1.0, _safe_float(static_cfg.get("score_saturation_mm"), 40.0))
    if static_status == "clear":
        static_clearances = [
            value for value in (
                full_static.get("box_minimum_safety_clearance_mm"),
                full_static.get("neighbor_minimum_clearance_mm"),
            )
            if value is not None
        ]
        static_score = (
            float(np.clip(min(float(value) for value in static_clearances) / static_saturation, 0.0, 1.0))
            if static_clearances else 1.0
        )
    elif static_status == "warning":
        static_score = 0.25
    elif static_status == "disabled":
        static_score = 0.5
    else:
        static_score = 0.0
    score = (1.0 - static_weight) * score + static_weight * 100.0 * static_score

    motion_weight = float(np.clip(_safe_float(motion_cfg.get("score_weight"), 0.20), 0.0, 1.0))
    motion_saturation = max(1.0, _safe_float(motion_cfg.get("score_saturation_mm"), 40.0))
    if motion_status == "clear":
        motion_clearances = [
            value for value in (
                full_motion.get("box_minimum_safety_clearance_mm"),
                full_motion.get("neighbor_minimum_clearance_mm"),
            )
            if value is not None
        ]
        motion_score = (
            float(np.clip(min(float(value) for value in motion_clearances) / motion_saturation, 0.0, 1.0))
            if motion_clearances else 1.0
        )
    elif motion_status == "warning":
        motion_score = 0.25
    elif motion_status == "disabled":
        motion_score = 0.5
    else:
        motion_score = 0.0
    score = (1.0 - motion_weight) * score + motion_weight * 100.0 * motion_score
    valid = len(reasons) == 0
    if not valid:
        score *= 0.25

    result: Dict[str, Any] = {
        **dict(clock),
        "evaluation_stage": "full",
        "full_evaluated": True,
        "light_valid": True,
        "valid": bool(valid),
        "score": float(score),
        "warnings": warnings,
        "rejection_reasons": reasons,
        "inner_boundary_uv": _json_uv(inner_uv),
        "outer_boundary_uv": _json_uv(outer_uv),
        "inner_boundary_camera_mm": _json_vector(inner_camera),
        "outer_boundary_camera_mm": _json_vector(outer_camera),
        "rim_plane_midpoint_camera_mm": _json_vector(front_midpoint),
        "grasp_center_camera_mm": _json_vector(grasp_center),
        "pregrasp_center_camera_mm": _json_vector(pregrasp),
        "inner_contact_camera_mm": _json_vector(inner_contact),
        "outer_contact_camera_mm": _json_vector(outer_contact),
        "finger_path_centers_camera_mm": {
            "inner_pregrasp": _json_vector(inner_pre_center),
            "outer_pregrasp": _json_vector(outer_pre_center),
            "inner_rim": _json_vector(inner_front_center),
            "outer_rim": _json_vector(outer_front_center),
            "inner_open_inserted": _json_vector(inner_final_center),
            "outer_open_inserted": _json_vector(outer_final_center),
            "inner_closed_rim": _json_vector(inner_closed_front_center),
            "outer_closed_rim": _json_vector(outer_closed_front_center),
            "inner_closed": _json_vector(inner_closed_center),
            "outer_closed": _json_vector(outer_closed_center),
        },
        "approach_vector_camera": _json_vector(approach),
        "closing_axis_camera": _json_vector(closing_axis),
        "lateral_axis_camera": _json_vector(tangent),
        "wall_thickness_mm": float(wall_thickness),
        "desired_target_closing_gap_mm": float(desired_target_gap),
        "target_closing_gap_mm": float(target_gap),
        "desired_wall_compression_each_side_mm": float(desired_compression),
        "actual_wall_compression_each_side_mm": float(actual_compression),
        "approach_opening_mm": float(approach_opening),
        "opening_margin_mm": float(opening_margin),
        "rim_insert_depth_mm": float(rim_insert),
        "inner_finger_mouth_containment": float(inner_containment),
        "other_ring_overlap_ratio": float(neighbor_2d_overlap),
        "neighbor_2d_overlap_mode": overlap_mode,
        "neighbor_2d_clearance_mm": float(neighbor_2d_clearance_mm),
        "neighbor_2d_clearance_mode": clearance_mode,
        "neighbor_3d": neighbor_3d,
        "neighbor_3d_status": neighbor_3d_status,
        "neighbor_3d_clearance_mm": neighbor_3d.get("minimum_clearance_mm"),
        "full_gripper_static": full_static,
        "full_gripper_static_status": static_status,
        "full_gripper_static_box_status": static_box_status,
        "full_gripper_static_neighbor_status": static_neighbor_status,
        "full_gripper_motion": full_motion,
        "full_gripper_motion_status": motion_status,
        "full_gripper_motion_box_status": motion_box_status,
        "full_gripper_motion_neighbor_status": motion_neighbor_status,
        # Backward-compatible summary field now uses the 3-D value when known.
        "neighbor_clearance_mm": (
            float(neighbor_3d.get("minimum_clearance_mm"))
            if neighbor_3d.get("minimum_clearance_mm") is not None
            else float(neighbor_2d_clearance_mm)
        ),
        "image_border_margin_px": float(border_margin),
        "box_wall": box_wall,
        "box_wall_clearance_mm": box_wall.get("clearance_mm"),
        "box_wall_status": box_status,
        "local_front_obstacle": front_obstacle,
        "inner_finger_sweep_polygon_uv": inner_polygon.reshape(-1, 2).astype(float).tolist() if inner_polygon is not None else None,
        "outer_finger_sweep_polygon_uv": outer_polygon.reshape(-1, 2).astype(float).tolist() if outer_polygon is not None else None,
        "outer_boundary_ambiguous": bool(outer_info.get("ambiguous")),
        "outer_boundary_source": outer_boundary_source,
    }
    result["grasp_frame_camera"] = _robot_grasp_frame(result, config)
    timing_ms["total_ms"] = _elapsed_ms(candidate_started)
    result["timing_ms"] = timing_ms
    return result


def analyze_ring_pair(
    ring: SegmentationInstance,
    mouth: SegmentationInstance,
    association: Dict[str, Any],
    all_rings: Sequence[SegmentationInstance],
    ring_mouth_masks: Mapping[int, np.ndarray],
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    config: GeometryConfig,
    *,
    evaluation_mode: str = "exhaustive",
    neighbor_base_cache: Optional[Mapping[int, Mapping[str, Any]]] = None,
    clock_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    pair_started = time.perf_counter()
    timing_ms: Dict[str, float] = {}
    resolved_evaluation_mode = str(evaluation_mode).strip().lower()
    staged = resolved_evaluation_mode in {"staged", "first_valid"}
    optimization = _optimization_settings(config)
    quality_cfg = config.section("quality")
    depth_cfg = config.section("depth")
    plane_cfg = config.section("plane")
    gripper_cfg = config.section("gripper")
    reasons: List[str] = []
    warnings: List[str] = []
    ring_area = ring.area_px
    mouth_area = mouth.area_px
    if ring_area < _safe_int(quality_cfg.get("minimum_ring_area_px"), 300):
        reasons.append("ring_area_too_small")
    if mouth_area < _safe_int(quality_cfg.get("minimum_mouth_area_px"), 80):
        reasons.append("mouth_area_too_small")

    ellipse_started = time.perf_counter()
    ellipse = fit_mouth_ellipse(mouth.mask)
    timing_ms["ellipse_fit_ms"] = _elapsed_ms(ellipse_started)
    if ellipse is None or len(ellipse.get("contour", [])) < _safe_int(quality_cfg.get("minimum_ellipse_points"), 12):
        reasons.append("mouth_ellipse_unavailable")
        timing_ms["total_ms"] = _elapsed_ms(pair_started)
        return {
            "ring_instance_id": ring.instance_id,
            "mouth_instance_id": mouth.instance_id,
            "ring_confidence": float(ring.confidence),
            "mouth_confidence": float(mouth.confidence),
            "association": association,
            "eligible": False,
            "robot_ready": False,
            "warnings": warnings,
            "rejection_reasons": reasons,
            "grasp": {"clock_candidates": [], "best_clock_candidate": None},
            "timing_ms": timing_ms,
        }

    depth_points_started = time.perf_counter()
    equivalent_radius = math.sqrt(max(1.0, float(mouth_area)) / math.pi)
    expand = int(round(equivalent_radius * _safe_float(depth_cfg.get("front_band_expand_ratio"), 0.40)))
    expand = max(_safe_int(depth_cfg.get("minimum_front_band_px"), 6), expand)
    expand = min(_safe_int(depth_cfg.get("maximum_front_band_px"), 26), expand)
    exclusion = _safe_int(depth_cfg.get("mouth_exclusion_px"), 2)
    ring_erode = _safe_int(depth_cfg.get("mask_erode_px"), 2)
    front_band = _erode(ring.mask, ring_erode) & _dilate(mouth.mask, expand) & ~_dilate(mouth.mask, exclusion)
    minimum_depth = _safe_float(depth_cfg.get("minimum_mm"), 150.0)
    maximum_depth = _safe_float(depth_cfg.get("maximum_mm"), 3000.0)
    points, pixels = depth_pixels_to_points(depth, front_band, intrinsics, minimum_depth, maximum_depth)
    band_pixel_count = int(np.count_nonzero(front_band))
    valid_ratio = float(len(points)) / float(max(1, band_pixel_count))
    timing_ms["front_depth_points_ms"] = _elapsed_ms(depth_points_started)
    if len(points) < _safe_int(depth_cfg.get("minimum_valid_points"), 80):
        reasons.append("insufficient_front_pose_depth")
    if valid_ratio < _safe_float(quality_cfg.get("minimum_depth_valid_ratio"), 0.32):
        reasons.append("low_front_pose_depth_valid_ratio")

    plane_started = time.perf_counter()
    depth_plane = fit_plane_ransac(points, config) if len(points) >= 3 else None
    timing_ms["plane_ransac_ms"] = _elapsed_ms(plane_started)
    if depth_plane is None:
        reasons.append("front_pose_fit_failed")
        timing_ms["total_ms"] = _elapsed_ms(pair_started)
        return {
            "ring_instance_id": ring.instance_id,
            "mouth_instance_id": mouth.instance_id,
            "ring_confidence": float(ring.confidence),
            "mouth_confidence": float(mouth.confidence),
            "association": association,
            "front_band_pixel_count": band_pixel_count,
            "front_plane_point_count": int(len(points)),
            "depth_valid_ratio": valid_ratio,
            "eligible": False,
            "robot_ready": False,
            "warnings": warnings,
            "rejection_reasons": reasons,
            "grasp": {"clock_candidates": [], "best_clock_candidate": None},
            "timing_ms": timing_ms,
        }
    if depth_plane.inlier_ratio < _safe_float(plane_cfg.get("minimum_inlier_ratio"), 0.34):
        reasons.append("low_plane_inlier_ratio")

    pose_started = time.perf_counter()
    pose_plane, pose_diagnostics = build_pose_plane(ellipse, intrinsics, points, depth_plane, config)
    pose_cfg = config.section("pose")
    disagreement = float(pose_diagnostics.get("normal_disagreement_deg", 0.0))
    disagreement_warn = _safe_float(pose_cfg.get("normal_disagreement_warning_deg"), 20.0)
    if disagreement > disagreement_warn:
        warnings.append("depth_plane_ellipse_pose_disagreement")

    # M37.5 safety gate: an ellipse-derived normal and a depth plane that
    # disagree by tens of degrees do not define a trustworthy grasp axis.
    # Reject the pair instead of forcing one source to win.
    if bool(pose_cfg.get("pose_conflict_hard_reject_enabled", True)):
        hard_limit = _safe_float(pose_cfg.get("maximum_normal_disagreement_deg"), 25.0)
        if disagreement > hard_limit:
            reasons.append("depth_plane_ellipse_pose_conflict")
        conditional_limit = _safe_float(
            pose_cfg.get("conditional_normal_disagreement_deg"), 18.0
        )
        minimum_depth_support = _safe_float(
            pose_cfg.get("conditional_minimum_depth_plane_inlier_ratio"), 0.55
        )
        if (
            str(pose_diagnostics.get("normal_source")) == "ellipse_stabilized"
            and disagreement > conditional_limit
            and depth_plane.inlier_ratio < minimum_depth_support
        ):
            reasons.append("ellipse_pose_has_insufficient_depth_support")
        stabilized_p95 = pose_diagnostics.get("stabilized_residual_p95_mm")
        if (
            stabilized_p95 is not None
            and float(stabilized_p95)
            > _safe_float(pose_cfg.get("maximum_stabilized_residual_p95_mm"), 8.0)
        ):
            reasons.append("ellipse_stabilized_pose_residual_too_high")

    center_uv = tuple(ellipse["center_uv"])
    center_camera = ray_plane_intersection(center_uv, intrinsics, pose_plane)
    if center_camera is None:
        center_camera = pose_plane.centroid.copy()
        warnings.append("mouth_center_fallback_to_pose_centroid")

    major_metric = _axis_metric_on_plane(ellipse, intrinsics, pose_plane, major=True)
    minor_metric = _axis_metric_on_plane(ellipse, intrinsics, pose_plane, major=False)
    if major_metric is None or minor_metric is None:
        reasons.append("mouth_axis_metric_failed")

    camera_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    tilt_cos = float(np.clip(abs(np.dot(pose_plane.normal, camera_axis)), 0.0, 1.0))
    tilt_deg = math.degrees(math.acos(tilt_cos))
    candidate_tilt_limit = _safe_float(gripper_cfg.get("geometry_candidate_max_tilt_deg"), 45.0)
    if tilt_deg > candidate_tilt_limit:
        reasons.append("tilt_exceeds_geometry_candidate_limit")

    mouth_major_mm = float(major_metric["length_mm"]) if major_metric else None
    mouth_minor_mm = float(minor_metric["length_mm"]) if minor_metric else None
    object_cfg = config.section("object_geometry")
    if mouth_major_mm is not None and mouth_minor_mm is not None:
        mean_diameter = 0.5 * (mouth_major_mm + mouth_minor_mm)
        minimum_diameter = _safe_float(object_cfg.get("minimum_inner_diameter_mm"), 35.0)
        maximum_diameter = _safe_float(object_cfg.get("maximum_inner_diameter_mm"), 100.0)
        if not (minimum_diameter <= mean_diameter <= maximum_diameter):
            if bool(object_cfg.get("physical_size_hard_reject", False)):
                reasons.append("mouth_physical_size_out_of_range")
            else:
                warnings.append("mouth_physical_size_out_of_range")
    timing_ms["pose_and_size_ms"] = _elapsed_ms(pose_started)

    mask_started = time.perf_counter()
    other_ring_mask = np.zeros(ring.mask.shape, dtype=bool)
    for item in all_rings:
        if int(item.instance_id) != int(ring.instance_id):
            other_ring_mask |= item.mask.astype(bool)
    other_ring_distance_map = _distance_map_to_mask_px(other_ring_mask)
    timing_ms["other_ring_mask_cache_ms"] = _elapsed_ms(mask_started)

    neighbor_clouds: List[Dict[str, Any]] = []
    neighbor_cloud_summary: Dict[str, Any] = {
        "enabled": bool(config.section("neighbor_3d").get("enabled", True)),
        "status": "deferred_by_staged_evaluation" if staged else "pending",
        "neighbor_instance_count": max(0, len(all_rings) - 1),
        "ready_instance_count": 0,
        "retained_point_count": 0,
        "instances": [],
    }
    if not staged:
        neighbor_started = time.perf_counter()
        neighbor_clouds, neighbor_cloud_summary = _prepare_neighbor_point_clouds(
            all_rings,
            ring,
            ring_mouth_masks,
            depth,
            intrinsics,
            pose_plane,
            config,
            base_cache=neighbor_base_cache,
        )
        timing_ms["neighbor_cloud_prepare_ms"] = _elapsed_ms(neighbor_started)

    count = _safe_int(gripper_cfg.get("clock_position_count"), 12)
    clocks = [dict(row) for row in clock_rows] if clock_rows is not None else _clock_positions(count)
    candidate_started = time.perf_counter()
    if staged and bool(optimization.get("skip_rejected_pairs", True)) and reasons:
        clock_candidates: List[Dict[str, Any]] = []
    else:
        clock_candidates = [
            _clock_candidate(
                clock,
                ring,
                mouth,
                other_ring_mask,
                neighbor_clouds,
                depth,
                intrinsics,
                pose_plane,
                center_camera,
                tilt_deg,
                config,
                evaluation_level="light" if staged else "full",
                other_ring_distance_map=other_ring_distance_map,
            )
            for clock in clocks
        ]
    timing_ms["clock_candidates_initial_ms"] = _elapsed_ms(candidate_started)

    if staged:
        light_valid = [item for item in clock_candidates if item.get("light_valid")]
        light_valid.sort(key=lambda item: float(item.get("light_score", item.get("score", 0.0))), reverse=True)
        best = None
        best_light = light_valid[0] if light_valid else None
    else:
        valid_candidates = [item for item in clock_candidates if item.get("valid")]
        valid_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        best = valid_candidates[0] if valid_candidates else None
        best_light = None
        if best is None:
            reasons.append("no_valid_rim_pinch_clock_position")

    normal = pose_plane.normal
    approach = -normal
    result: Dict[str, Any] = {
        "ring_instance_id": int(ring.instance_id),
        "mouth_instance_id": int(mouth.instance_id),
        "ring_confidence": float(ring.confidence),
        "mouth_confidence": float(mouth.confidence),
        "association": association,
        "ring_area_px": int(ring_area),
        "mouth_area_px": int(mouth_area),
        "front_band_pixel_count": band_pixel_count,
        "front_plane_point_count": int(len(points)),
        "depth_valid_ratio": float(valid_ratio),
        "mouth_ellipse": {
            "center_uv": _json_uv(center_uv),
            "major_px": float(ellipse["major_px"]),
            "minor_px": float(ellipse["minor_px"]),
            "angle_deg": float(ellipse["angle_deg"]),
        },
        "plane": {
            "normal_toward_camera": _json_vector(depth_plane.normal),
            "offset": float(depth_plane.offset),
            "centroid_camera_mm": _json_vector(depth_plane.centroid),
            "inlier_ratio": float(depth_plane.inlier_ratio),
            "residual_median_mm": float(depth_plane.residual_median_mm),
            "residual_p95_mm": float(depth_plane.residual_p95_mm),
        },
        "pose": {
            **pose_diagnostics,
            "normal_toward_camera": _json_vector(pose_plane.normal),
            "offset": float(pose_plane.offset),
            "centroid_camera_mm": _json_vector(pose_plane.centroid),
            "inlier_ratio": float(pose_plane.inlier_ratio),
            "residual_median_mm": float(pose_plane.residual_median_mm),
            "residual_p95_mm": float(pose_plane.residual_p95_mm),
        },
        "ring_center_camera_mm": _json_vector(center_camera),
        "ring_axis_toward_camera": _json_vector(normal),
        "approach_vector_camera": _json_vector(approach),
        "tilt_deg": float(tilt_deg),
        "neighbor_3d_point_clouds": neighbor_cloud_summary,
        "mouth_major_mm": mouth_major_mm,
        "mouth_minor_mm": mouth_minor_mm,
        "grasp": {
            "mode": "rim_pinch",
            "description": "one finger inside, one finger outside, then close on the local foam wall",
            "selection_scope": (
                (
                    "first_valid_adaptive_light_then_complete_collision"
                    if resolved_evaluation_mode == "first_valid" else
                    "staged_light_then_budgeted_complete_3d_collision"
                )
                if staged else
                "12_clock_rim_pinch_with_3d_box_depth_neighbor_complete_gripper_static_and_pregrasp_motion"
            ),
            "clock_position_count": int(count),
            "generated_clock_candidate_count": int(len(clocks)),
            "best_clock_candidate": best,
            "best_light_clock_candidate": best_light,
            "clock_candidates": clock_candidates,
        },
        "eligible": bool(not staged and len(reasons) == 0 and best is not None),
        "robot_ready": False,
        "warnings": warnings,
        "rejection_reasons": list(reasons),
        "timing_ms": timing_ms,
        "_debug": {
            "front_band_mask": front_band,
            "plane_points": points,
            "plane_pixels": pixels,
            "plane_inlier_mask": depth_plane.inlier_mask,
        },
    }
    if staged:
        result["candidate_evaluation"] = {
            "mode": resolved_evaluation_mode,
            "light_candidate_count": len(clock_candidates),
            "light_valid_count": len([item for item in clock_candidates if item.get("light_valid")]),
            "full_evaluated_count": 0,
            "full_valid_count": 0,
            "deferred_count": len(clock_candidates),
        }
        result["_optimization_context"] = {
            "ring": ring,
            "mouth": mouth,
            "other_ring_mask": other_ring_mask,
            "other_ring_distance_map": other_ring_distance_map,
            "depth": depth,
            "intrinsics": intrinsics,
            "pose_plane": pose_plane,
            "center_camera": center_camera,
            "tilt_deg": float(tilt_deg),
            "all_rings": all_rings,
            "ring_mouth_masks": ring_mouth_masks,
            "neighbor_clouds": None,
            "neighbor_cloud_summary": None,
        }
    else:
        diagnostic_warnings = {"neighbor_2d_overlap_warning", "neighbor_2d_clearance_warning"}
        best_blocking_warnings = [
            warning for warning in (best.get("warnings") if best else [])
            if str(warning) not in diagnostic_warnings
        ]
        result["initial_robot_safe_geometry"] = bool(
            result["eligible"]
            and tilt_deg <= _safe_float(gripper_cfg.get("robot_safe_max_tilt_deg"), 30.0)
            and not best_blocking_warnings
            and str((best.get("neighbor_3d") or {}).get("status") or "unknown") == "clear"
            and str((best.get("full_gripper_static") or {}).get("status") or "unknown") == "clear"
            and str((best.get("full_gripper_motion") or {}).get("status") or "unknown") == "clear"
        )
    timing_ms["total_ms"] = _elapsed_ms(pair_started)
    return result


def _finalize_staged_pair(result: Dict[str, Any], config: GeometryConfig) -> None:
    grasp = result.get("grasp") if isinstance(result.get("grasp"), dict) else {}
    candidates = grasp.get("clock_candidates") if isinstance(grasp.get("clock_candidates"), list) else []
    full_candidates = [item for item in candidates if item.get("full_evaluated")]
    valid_candidates = [item for item in full_candidates if item.get("valid")]
    valid_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = valid_candidates[0] if valid_candidates else None
    grasp["best_clock_candidate"] = best
    result["grasp"] = grasp
    result["eligible"] = bool(len(result.get("rejection_reasons") or []) == 0 and best is not None)
    diagnostic_warnings = {"neighbor_2d_overlap_warning", "neighbor_2d_clearance_warning"}
    best_blocking_warnings = [
        warning for warning in (best.get("warnings") if best else [])
        if str(warning) not in diagnostic_warnings
    ]
    result["initial_robot_safe_geometry"] = bool(
        result["eligible"]
        and float(result.get("tilt_deg", 180.0)) <= _safe_float(config.section("gripper").get("robot_safe_max_tilt_deg"), 30.0)
        and not best_blocking_warnings
        and str((best.get("neighbor_3d") or {}).get("status") or "unknown") == "clear"
        and str((best.get("full_gripper_static") or {}).get("status") or "unknown") == "clear"
        and str((best.get("full_gripper_motion") or {}).get("status") or "unknown") == "clear"
    )
    evaluation = result.get("candidate_evaluation") if isinstance(result.get("candidate_evaluation"), dict) else {}
    evaluation.update({
        "full_evaluated_count": len(full_candidates),
        "full_valid_count": len(valid_candidates),
        "deferred_count": len([item for item in candidates if not item.get("full_evaluated")]),
        "status": (
            "valid_found"
            if valid_candidates else
            ("no_valid_within_budget" if full_candidates else "not_selected_for_full_evaluation")
        ),
    })
    result["candidate_evaluation"] = evaluation


def analyze_scene(
    instances: Sequence[SegmentationInstance],
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    config: GeometryConfig,
) -> Dict[str, Any]:
    scene_started = time.perf_counter()
    scene_timing: Dict[str, float] = {}
    optimization = _optimization_settings(config)
    mode = str(optimization.get("mode") or "exhaustive")
    staged = bool(optimization.get("enabled")) and mode == "staged"
    first_valid = bool(optimization.get("enabled")) and mode == "first_valid"
    classes = config.section("classes")
    ring_name = str(classes.get("foam_ring") or "foam_ring")
    mouth_name = str(classes.get("ring_mouth") or "ring_mouth")
    rings = [item for item in instances if item.class_name == ring_name]
    mouths = [item for item in instances if item.class_name == mouth_name]
    association_started = time.perf_counter()
    matches, unmatched_rings, unmatched_mouths, association_debug = _associate_ring_mouths_detailed(
        rings,
        mouths,
        config,
    )
    all_matches = list(matches)
    candidate_scope = config.section("candidate_scope")
    allowed_ring_ids_raw = candidate_scope.get("allowed_ring_instance_ids")
    candidate_scope_enabled = isinstance(allowed_ring_ids_raw, Sequence) and not isinstance(
        allowed_ring_ids_raw, (str, bytes)
    )
    allowed_ring_ids = (
        {int(value) for value in allowed_ring_ids_raw}
        if candidate_scope_enabled
        else None
    )
    if allowed_ring_ids is not None:
        matches = [
            row for row in all_matches
            if int(row[0].instance_id) in allowed_ring_ids
        ]
    scene_timing["association_ms"] = _elapsed_ms(association_started)
    # Keep masks for every globally associated ring so active-layer M36
    # candidates still see all neighboring objects during collision checks.
    ring_mouth_masks = {
        int(ring.instance_id): mouth.mask for ring, mouth, _ in all_matches
    }
    results: List[Dict[str, Any]] = []
    optimization_summary: Dict[str, Any] = {
        **optimization,
        "matched_pair_count": len(matches),
        "light_candidate_count": 0,
        "light_valid_count": 0,
        "full_candidate_evaluated_count": 0,
        "full_candidate_valid_count": 0,
        "candidate_budget_exhausted": False,
        "neighbor_base_cache": None,
        "pair_preselection_count": 0,
        "fully_analyzed_pair_count": 0,
        "deferred_pair_count": 0,
        "adaptive_fallback_used": False,
        "primary_light_candidate_count": 0,
        "fallback_light_candidate_count": 0,
        "first_valid_pair_rank": None,
        "first_valid_candidate_search_batch": None,
        "first_valid_candidate_clock_hour": None,
        "early_exit_triggered": False,
    }

    if first_valid:
        # M36.4.2: rank pairs using only masks, association metrics and sparse
        # depth samples. No point-cloud construction or RANSAC is performed here.
        preselection_started = time.perf_counter()
        ranked_pairs: List[Dict[str, Any]] = []
        for ring, mouth, metrics in matches:
            preselection = _pair_preselection_metrics(ring, mouth, metrics, depth, config)
            ranked_pairs.append({
                "ring": ring,
                "mouth": mouth,
                "association": metrics,
                "preselection": preselection,
                "rank_key": _pair_preselection_rank(
                    preselection,
                    bool(optimization.get("prefer_top_layer", True)),
                ),
            })
        ranked_pairs.sort(key=lambda row: row["rank_key"], reverse=True)
        for rank, row in enumerate(ranked_pairs, start=1):
            row["preselection"]["rank"] = int(rank)
        scene_timing["pair_preselection_ms"] = _elapsed_ms(preselection_started)
        optimization_summary["pair_preselection_count"] = len(ranked_pairs)

        primary_clocks, fallback_clocks = _adaptive_clock_batches(config)
        maximum_pairs = int(optimization["maximum_pairs_to_fully_analyze"])
        maximum_candidates_per_pair = int(optimization["maximum_full_candidates_per_pair"])
        base_cache: Optional[Dict[int, Dict[str, Any]]] = None
        base_cache_summary: Optional[Dict[str, Any]] = None
        base_cache_ms = 0.0
        pair_initial_ms = 0.0
        full_evaluation_ms = 0.0
        fallback_light_ms = 0.0
        analyzed_keys: set[Tuple[int, int]] = set()
        selected_found = False

        for row in ranked_pairs:
            rank = int(row["preselection"]["rank"])
            if rank > maximum_pairs:
                optimization_summary["candidate_budget_exhausted"] = True
                break
            ring = row["ring"]
            mouth = row["mouth"]
            metrics = row["association"]
            pair_key = (int(ring.instance_id), int(mouth.instance_id))
            analyzed_keys.add(pair_key)

            pair_started = time.perf_counter()
            pair_result = analyze_ring_pair(
                ring,
                mouth,
                metrics,
                rings,
                ring_mouth_masks,
                depth,
                intrinsics,
                config,
                evaluation_mode="first_valid",
                clock_rows=primary_clocks,
            )
            pair_initial_ms += _elapsed_ms(pair_started)
            pair_result["pair_preselection"] = dict(row["preselection"])
            pair_result["processing_status"] = "analyzed"
            results.append(pair_result)
            optimization_summary["fully_analyzed_pair_count"] += 1

            context = pair_result.get("_optimization_context")
            candidates = ((pair_result.get("grasp") or {}).get("clock_candidates") or [])
            optimization_summary["primary_light_candidate_count"] += len(candidates)
            optimization_summary["light_candidate_count"] += len(candidates)
            optimization_summary["light_valid_count"] += len(
                [candidate for candidate in candidates if bool(candidate.get("light_valid"))]
            )

            valid_found = False
            evaluated_for_pair = 0

            def ensure_neighbor_clouds() -> bool:
                nonlocal base_cache, base_cache_summary, base_cache_ms, full_evaluation_ms
                if not isinstance(context, dict):
                    return False
                if context.get("neighbor_clouds") is not None:
                    return True
                if base_cache is None and bool(optimization.get("cache_neighbor_point_clouds")):
                    cache_started = time.perf_counter()
                    base_cache, base_cache_summary = _build_neighbor_base_cache(
                        rings,
                        ring_mouth_masks,
                        depth,
                        intrinsics,
                        config,
                    )
                    elapsed = _elapsed_ms(cache_started)
                    base_cache_ms += elapsed
                    optimization_summary["neighbor_base_cache"] = base_cache_summary
                neighbor_stage_started = time.perf_counter()
                neighbor_started = time.perf_counter()
                neighbor_clouds, neighbor_summary = _prepare_neighbor_point_clouds(
                    context["all_rings"],
                    context["ring"],
                    context["ring_mouth_masks"],
                    context["depth"],
                    context["intrinsics"],
                    context["pose_plane"],
                    config,
                    base_cache=base_cache,
                )
                context["neighbor_clouds"] = neighbor_clouds
                context["neighbor_cloud_summary"] = neighbor_summary
                pair_result["neighbor_3d_point_clouds"] = neighbor_summary
                pair_timing = pair_result.get("timing_ms") if isinstance(pair_result.get("timing_ms"), dict) else {}
                pair_timing["neighbor_cloud_prepare_ms"] = _elapsed_ms(neighbor_started)
                pair_result["timing_ms"] = pair_timing
                full_evaluation_ms += _elapsed_ms(neighbor_stage_started)
                return True

            def evaluate_candidate(candidate_index: int) -> bool:
                nonlocal evaluated_for_pair, full_evaluation_ms
                if not isinstance(context, dict):
                    return False
                if evaluated_for_pair >= maximum_candidates_per_pair:
                    optimization_summary["candidate_budget_exhausted"] = True
                    return False
                current_candidates = pair_result["grasp"]["clock_candidates"]
                original = current_candidates[candidate_index]
                if not bool(original.get("light_valid")):
                    return False
                if not ensure_neighbor_clouds():
                    return False
                candidate_started = time.perf_counter()
                full_candidate = _clock_candidate(
                    original,
                    context["ring"],
                    context["mouth"],
                    context["other_ring_mask"],
                    context["neighbor_clouds"],
                    context["depth"],
                    context["intrinsics"],
                    context["pose_plane"],
                    context["center_camera"],
                    float(context["tilt_deg"]),
                    config,
                    evaluation_level="full",
                    other_ring_distance_map=context["other_ring_distance_map"],
                )
                elapsed = _elapsed_ms(candidate_started)
                full_evaluation_ms += elapsed
                full_candidate["evaluation_stage"] = "full"
                full_candidate["full_evaluated"] = True
                full_candidate["light_score"] = original.get("light_score", original.get("score"))
                full_candidate["light_rank_source"] = "M36.4.2_first_valid_adaptive_clock_ranking"
                current_candidates[candidate_index] = full_candidate
                evaluated_for_pair += 1
                optimization_summary["full_candidate_evaluated_count"] += 1
                if bool(full_candidate.get("valid")):
                    optimization_summary["full_candidate_valid_count"] += 1
                    optimization_summary["first_valid_pair_rank"] = rank
                    optimization_summary["first_valid_candidate_search_batch"] = full_candidate.get("search_batch")
                    optimization_summary["first_valid_candidate_clock_hour"] = full_candidate.get("clock_hour")
                    return True
                return False

            if isinstance(context, dict) and (
                not bool(optimization.get("skip_rejected_pairs"))
                or not pair_result.get("rejection_reasons")
            ):
                primary_order = sorted(
                    [
                        index for index, candidate in enumerate(candidates)
                        if bool(candidate.get("light_valid"))
                    ],
                    key=lambda index: float(
                        candidates[index].get("light_score", candidates[index].get("score", 0.0))
                    ),
                    reverse=True,
                )
                for candidate_index in primary_order:
                    if evaluate_candidate(candidate_index):
                        valid_found = True
                        if bool(optimization.get("stop_after_first_valid_candidate", True)):
                            break

                if (
                    not valid_found
                    and fallback_clocks
                    and evaluated_for_pair < maximum_candidates_per_pair
                ):
                    fallback_started = time.perf_counter()
                    fallback_candidates = [
                        _clock_candidate(
                            clock,
                            context["ring"],
                            context["mouth"],
                            context["other_ring_mask"],
                            [],
                            context["depth"],
                            context["intrinsics"],
                            context["pose_plane"],
                            context["center_camera"],
                            float(context["tilt_deg"]),
                            config,
                            evaluation_level="light",
                            other_ring_distance_map=context["other_ring_distance_map"],
                        )
                        for clock in fallback_clocks
                    ]
                    fallback_light_ms += _elapsed_ms(fallback_started)
                    start_index = len(pair_result["grasp"]["clock_candidates"])
                    pair_result["grasp"]["clock_candidates"].extend(fallback_candidates)
                    pair_result["grasp"]["generated_clock_candidate_count"] = len(
                        pair_result["grasp"]["clock_candidates"]
                    )
                    optimization_summary["adaptive_fallback_used"] = True
                    optimization_summary["fallback_light_candidate_count"] += len(fallback_candidates)
                    optimization_summary["light_candidate_count"] += len(fallback_candidates)
                    optimization_summary["light_valid_count"] += len(
                        [candidate for candidate in fallback_candidates if bool(candidate.get("light_valid"))]
                    )
                    fallback_order = sorted(
                        [
                            start_index + offset
                            for offset, candidate in enumerate(fallback_candidates)
                            if bool(candidate.get("light_valid"))
                        ],
                        key=lambda index: float(
                            pair_result["grasp"]["clock_candidates"][index].get(
                                "light_score",
                                pair_result["grasp"]["clock_candidates"][index].get("score", 0.0),
                            )
                        ),
                        reverse=True,
                    )
                    for candidate_index in fallback_order:
                        if evaluate_candidate(candidate_index):
                            valid_found = True
                            if bool(optimization.get("stop_after_first_valid_candidate", True)):
                                break

            for candidate in ((pair_result.get("grasp") or {}).get("clock_candidates") or []):
                if not candidate.get("full_evaluated"):
                    candidate["evaluation_stage"] = "deferred"
                    candidate["deferred_reason"] = (
                        "first_valid_candidate_found"
                        if valid_found else
                        "not_reached_before_pair_candidate_limit"
                    )
            if isinstance(pair_result.get("_optimization_context"), Mapping):
                _finalize_staged_pair(pair_result, config)
                pair_result.pop("_optimization_context", None)
            pair_result["processing_status"] = (
                "selected_first_valid" if pair_result.get("eligible") else "analyzed_no_valid_grasp"
            )
            if pair_result.get("eligible"):
                selected_found = True
                if bool(optimization.get("stop_after_first_valid_target", True)):
                    optimization_summary["early_exit_triggered"] = True
                    break

        deferred_reason = (
            "after_first_valid_target"
            if selected_found else
            "maximum_pairs_to_fully_analyze_reached"
        )
        for row in ranked_pairs:
            ring = row["ring"]
            mouth = row["mouth"]
            key = (int(ring.instance_id), int(mouth.instance_id))
            if key in analyzed_keys:
                continue
            results.append(
                _deferred_pair_result(
                    ring,
                    mouth,
                    row["association"],
                    row["preselection"],
                    deferred_reason,
                )
            )
        optimization_summary["deferred_pair_count"] = len(ranked_pairs) - len(analyzed_keys)
        scene_timing["pair_geometry_initial_ms"] = float(pair_initial_ms)
        scene_timing["neighbor_base_cache_ms"] = float(base_cache_ms)
        scene_timing["adaptive_fallback_light_ms"] = float(fallback_light_ms)
        scene_timing["full_candidate_evaluation_ms"] = float(full_evaluation_ms)

    elif staged:
        pair_started = time.perf_counter()
        results = [
            analyze_ring_pair(
                ring,
                mouth,
                metrics,
                rings,
                ring_mouth_masks,
                depth,
                intrinsics,
                config,
                evaluation_mode="staged",
            )
            for ring, mouth, metrics in matches
        ]
        scene_timing["pair_geometry_initial_ms"] = _elapsed_ms(pair_started)
        scene_timing["pair_preselection_ms"] = 0.0
        scene_timing["adaptive_fallback_light_ms"] = 0.0
        candidate_refs: List[Tuple[Tuple[Any, ...], int, int]] = []
        pair_rows = [
            (index, item)
            for index, item in enumerate(results)
            if isinstance(item.get("_optimization_context"), Mapping)
            and item.get("ring_center_camera_mm")
            and (not bool(optimization.get("skip_rejected_pairs")) or not item.get("rejection_reasons"))
        ]
        optimization_summary["fully_analyzed_pair_count"] = len(results)
        nearest_pair_z = min(
            (float(item["ring_center_camera_mm"][2]) for _, item in pair_rows),
            default=None,
        )
        top_tolerance = _safe_float(config.section("gripper").get("top_layer_tolerance_mm"), 15.0)
        safe_tilt = _safe_float(config.section("gripper").get("robot_safe_max_tilt_deg"), 30.0)
        for pair_index, item in pair_rows:
            candidates = ((item.get("grasp") or {}).get("clock_candidates") or [])
            z_value = float(item["ring_center_camera_mm"][2])
            in_top_layer = nearest_pair_z is None or z_value <= nearest_pair_z + top_tolerance
            for candidate_index, candidate in enumerate(candidates):
                optimization_summary["light_candidate_count"] += 1
                if not bool(candidate.get("light_valid")):
                    continue
                optimization_summary["light_valid_count"] += 1
                rank = (
                    bool(in_top_layer) if bool(optimization.get("prefer_top_layer")) else True,
                    bool(float(item.get("tilt_deg", 180.0)) <= safe_tilt),
                    float(candidate.get("light_score", candidate.get("score", 0.0))),
                    0.5 * (float(item.get("ring_confidence", 0.0)) + float(item.get("mouth_confidence", 0.0))),
                    -float(item.get("tilt_deg", 180.0)),
                    -z_value,
                )
                candidate_refs.append((rank, pair_index, candidate_index))
        candidate_refs.sort(key=lambda row: row[0], reverse=True)
        if bool(optimization.get("round_robin_pairs")) and candidate_refs:
            grouped: Dict[int, List[Tuple[Tuple[Any, ...], int, int]]] = {}
            pair_order: List[int] = []
            for row in candidate_refs:
                pair_index = int(row[1])
                if pair_index not in grouped:
                    grouped[pair_index] = []
                    pair_order.append(pair_index)
                grouped[pair_index].append(row)
            round_robin: List[Tuple[Tuple[Any, ...], int, int]] = []
            offset = 0
            while True:
                appended = False
                for pair_index in pair_order:
                    rows = grouped[pair_index]
                    if offset < len(rows):
                        round_robin.append(rows[offset])
                        appended = True
                if not appended:
                    break
                offset += 1
            candidate_refs = round_robin

        base_cache: Optional[Dict[int, Dict[str, Any]]] = None
        if candidate_refs and bool(optimization.get("cache_neighbor_point_clouds")):
            cache_started = time.perf_counter()
            base_cache, cache_summary = _build_neighbor_base_cache(
                rings,
                ring_mouth_masks,
                depth,
                intrinsics,
                config,
            )
            scene_timing["neighbor_base_cache_ms"] = _elapsed_ms(cache_started)
            optimization_summary["neighbor_base_cache"] = cache_summary
        else:
            scene_timing["neighbor_base_cache_ms"] = 0.0

        full_started = time.perf_counter()
        initial_budget = int(optimization["initial_full_candidate_budget"])
        maximum_budget = int(optimization["maximum_full_candidate_budget"])
        minimum_valid = int(optimization["minimum_valid_full_candidates"])
        evaluated = 0
        valid_count = 0
        for _, pair_index, candidate_index in candidate_refs:
            if evaluated >= maximum_budget:
                optimization_summary["candidate_budget_exhausted"] = True
                break
            if evaluated >= initial_budget and valid_count >= minimum_valid:
                break
            if (
                evaluated >= initial_budget
                and valid_count == 0
                and not bool(optimization.get("expand_if_no_valid"))
            ):
                break
            pair_result = results[pair_index]
            context = pair_result.get("_optimization_context")
            if not isinstance(context, dict):
                continue
            if context.get("neighbor_clouds") is None:
                neighbor_started = time.perf_counter()
                neighbor_clouds, neighbor_summary = _prepare_neighbor_point_clouds(
                    context["all_rings"],
                    context["ring"],
                    context["ring_mouth_masks"],
                    context["depth"],
                    context["intrinsics"],
                    context["pose_plane"],
                    config,
                    base_cache=base_cache,
                )
                context["neighbor_clouds"] = neighbor_clouds
                context["neighbor_cloud_summary"] = neighbor_summary
                pair_result["neighbor_3d_point_clouds"] = neighbor_summary
                pair_timing = pair_result.get("timing_ms") if isinstance(pair_result.get("timing_ms"), dict) else {}
                pair_timing["neighbor_cloud_prepare_ms"] = _elapsed_ms(neighbor_started)
                pair_result["timing_ms"] = pair_timing
            original = pair_result["grasp"]["clock_candidates"][candidate_index]
            full_candidate = _clock_candidate(
                original,
                context["ring"],
                context["mouth"],
                context["other_ring_mask"],
                context["neighbor_clouds"],
                context["depth"],
                context["intrinsics"],
                context["pose_plane"],
                context["center_camera"],
                float(context["tilt_deg"]),
                config,
                evaluation_level="full",
                other_ring_distance_map=context["other_ring_distance_map"],
            )
            full_candidate["evaluation_stage"] = "full"
            full_candidate["full_evaluated"] = True
            full_candidate["light_score"] = original.get("light_score", original.get("score"))
            full_candidate["light_rank_source"] = "M36.4.1_staged_candidate_ranking"
            pair_result["grasp"]["clock_candidates"][candidate_index] = full_candidate
            evaluated += 1
            if bool(full_candidate.get("valid")):
                valid_count += 1

        scene_timing["full_candidate_evaluation_ms"] = _elapsed_ms(full_started)
        optimization_summary["full_candidate_evaluated_count"] = evaluated
        optimization_summary["full_candidate_valid_count"] = valid_count
        for result in results:
            candidates = ((result.get("grasp") or {}).get("clock_candidates") or [])
            for candidate in candidates:
                if not candidate.get("full_evaluated"):
                    candidate["evaluation_stage"] = "deferred"
                    candidate["deferred_reason"] = "outside_full_candidate_budget"
            if isinstance(result.get("_optimization_context"), Mapping):
                _finalize_staged_pair(result, config)
                result.pop("_optimization_context", None)
    else:
        pair_started = time.perf_counter()
        results = [
            analyze_ring_pair(
                ring,
                mouth,
                metrics,
                rings,
                ring_mouth_masks,
                depth,
                intrinsics,
                config,
                evaluation_mode="exhaustive",
            )
            for ring, mouth, metrics in matches
        ]
        scene_timing["pair_geometry_initial_ms"] = _elapsed_ms(pair_started)
        scene_timing["pair_preselection_ms"] = 0.0
        scene_timing["adaptive_fallback_light_ms"] = 0.0
        scene_timing["neighbor_base_cache_ms"] = 0.0
        scene_timing["full_candidate_evaluation_ms"] = 0.0
        optimization_summary["fully_analyzed_pair_count"] = len(results)
        exhaustive_candidates = [
            candidate
            for item in results
            for candidate in (((item.get("grasp") or {}).get("clock_candidates") or []))
        ]
        optimization_summary["full_candidate_evaluated_count"] = len(
            [candidate for candidate in exhaustive_candidates if candidate.get("full_evaluated")]
        )
        optimization_summary["full_candidate_valid_count"] = len(
            [candidate for candidate in exhaustive_candidates if candidate.get("valid")]
        )

    selection_started = time.perf_counter()
    eligible = [item for item in results if item.get("eligible") and item.get("ring_center_camera_mm")]
    selected_ring_id: Optional[int] = None
    selected_clock: Optional[int] = None
    selected_clock_angle: Optional[float] = None
    selected_clock_search_batch: Optional[str] = None
    selected_robot_candidate: Optional[Dict[str, Any]] = None
    if eligible:
        nearest_z = min(float(item["ring_center_camera_mm"][2]) for item in eligible)
        tolerance = _safe_float(config.section("gripper").get("top_layer_tolerance_mm"), 15.0)
        top = [item for item in eligible if float(item["ring_center_camera_mm"][2]) <= nearest_z + tolerance]
        safe_tilt = _safe_float(config.section("gripper").get("robot_safe_max_tilt_deg"), 30.0)
        top.sort(
            key=lambda item: (
                bool(float(item.get("tilt_deg", 180.0)) <= safe_tilt),
                float(((item.get("grasp") or {}).get("best_clock_candidate") or {}).get("score", 0.0)),
                -float(item.get("tilt_deg", 180.0)),
                0.5 * (float(item.get("ring_confidence", 0.0)) + float(item.get("mouth_confidence", 0.0))),
                -float(item["ring_center_camera_mm"][2]),
            ),
            reverse=True,
        )
        selected = top[0]
        best = (selected.get("grasp") or {}).get("best_clock_candidate") or {}
        selected["selected"] = True
        selected["selection_reason"] = {
            "top_layer_nearest_z_mm": float(nearest_z),
            "top_layer_tolerance_mm": float(tolerance),
            "priority": [
                "within_initial_robot_safe_tilt",
                "clock_candidate_score",
                "lower_tilt",
                "segmentation_confidence",
                "nearest_z",
            ],
        }
        selected_ring_id = int(selected["ring_instance_id"])
        selected_clock = int(best.get("clock_hour")) if best.get("clock_hour") is not None else None
        selected_clock_angle = (
            float(best.get("clock_angle_deg_cw_from_12"))
            if best.get("clock_angle_deg_cw_from_12") is not None else None
        )
        selected_clock_search_batch = (
            str(best.get("search_batch")) if best.get("search_batch") is not None else None
        )
        selected_robot_candidate = {
            "schema_version": "1.0",
            "message_type": "foam_ring_rim_pinch_grasp_candidate",
            "status": "candidate_only_not_robot_ready",
            "robot_ready": False,
            "reason": "M35.2_pregrasp_to_grasp_motion_checked; post_grasp_lift_intentionally_not_checked; robot_reachability_and_hand_eye_transform_not_applied",
            "target": {
                "ring_instance_id": selected_ring_id,
                "mouth_instance_id": int(selected["mouth_instance_id"]),
                "clock_hour": selected_clock,
                "clock_angle_deg_cw_from_12": best.get("clock_angle_deg_cw_from_12"),
                "clock_search_batch": best.get("search_batch"),
                "candidate_score": best.get("score"),
                "tilt_deg": selected.get("tilt_deg"),
                "box_wall_status": best.get("box_wall_status"),
                "box_wall_clearance_mm": best.get("box_wall_clearance_mm"),
                "box_wall_nearest_wall": (best.get("box_wall") or {}).get("nearest_wall"),
                "box_wall_worst_stage": (best.get("box_wall") or {}).get("worst_stage"),
                "neighbor_3d_status": (best.get("neighbor_3d") or {}).get("status"),
                "neighbor_3d_clearance_mm": (best.get("neighbor_3d") or {}).get("minimum_clearance_mm"),
                "neighbor_3d_nearest_instance_id": (best.get("neighbor_3d") or {}).get("nearest_instance_id"),
                "neighbor_3d_colliding_instance_ids": (best.get("neighbor_3d") or {}).get("colliding_instance_ids"),
                "neighbor_3d_worst_stage": (best.get("neighbor_3d") or {}).get("worst_stage"),
                "full_gripper_static_status": (best.get("full_gripper_static") or {}).get("status"),
                "full_gripper_static_box_status": (best.get("full_gripper_static") or {}).get("box_status"),
                "full_gripper_static_neighbor_status": (best.get("full_gripper_static") or {}).get("neighbor_status"),
                "full_gripper_static_box_clearance_mm": (best.get("full_gripper_static") or {}).get("box_minimum_clearance_mm"),
                "full_gripper_static_neighbor_clearance_mm": (best.get("full_gripper_static") or {}).get("neighbor_minimum_clearance_mm"),
                "full_gripper_static_worst_box_component": (best.get("full_gripper_static") or {}).get("box_worst_component"),
                "full_gripper_static_worst_neighbor_component": (best.get("full_gripper_static") or {}).get("neighbor_worst_component"),
                "full_gripper_motion_status": (best.get("full_gripper_motion") or {}).get("status"),
                "full_gripper_motion_box_status": (best.get("full_gripper_motion") or {}).get("box_status"),
                "full_gripper_motion_neighbor_status": (best.get("full_gripper_motion") or {}).get("neighbor_status"),
                "full_gripper_motion_worst_stage": (best.get("full_gripper_motion") or {}).get("worst_stage"),
                "full_gripper_motion_box_clearance_mm": (best.get("full_gripper_motion") or {}).get("box_minimum_clearance_mm"),
                "full_gripper_motion_neighbor_clearance_mm": (best.get("full_gripper_motion") or {}).get("neighbor_minimum_clearance_mm"),
            },
            "grasp_frame_camera": best.get("grasp_frame_camera"),
            "mounting_interface_frame_camera": (best.get("full_gripper_static") or {}).get("mounting_interface_frame_camera"),
            "pregrasp_center_camera_mm": best.get("pregrasp_center_camera_mm"),
            "rim_plane_midpoint_camera_mm": best.get("rim_plane_midpoint_camera_mm"),
            "inner_contact_camera_mm": best.get("inner_contact_camera_mm"),
            "outer_contact_camera_mm": best.get("outer_contact_camera_mm"),
            "pregrasp_motion": {
                "scope": "pregrasp_to_grasp_only",
                "post_grasp_lift_checked": False,
                "path_keyframes_camera": (best.get("full_gripper_motion") or {}).get("path_keyframes_camera"),
                "stage_summaries": (best.get("full_gripper_motion") or {}).get("stage_summaries"),
            },
            "gripper_command": {
                "travel_opening_mm": (best.get("full_gripper_motion") or {}).get("travel_opening_mm"),
                "open_start_offset_mm": (best.get("full_gripper_motion") or {}).get("open_start_offset_mm"),
                "opening_before_approach_mm": best.get("approach_opening_mm"),
                "target_closing_gap_mm": best.get("target_closing_gap_mm"),
                "rim_insert_depth_mm": best.get("rim_insert_depth_mm"),
                "desired_wall_compression_each_side_mm": best.get("desired_wall_compression_each_side_mm"),
                "actual_wall_compression_each_side_mm": best.get("actual_wall_compression_each_side_mm"),
            },
        }
    scene_timing["selection_ms"] = _elapsed_ms(selection_started)
    for item in results:
        item.setdefault("selected", False)
    box_summary_started = time.perf_counter()
    box_wall_model = _box_wall_model(depth.shape, config, intrinsics)
    scene_timing["box_model_summary_ms"] = _elapsed_ms(box_summary_started)
    scene_timing["total_ms"] = _elapsed_ms(scene_started)
    full_candidate_rows = [
        candidate
        for item in results
        for candidate in (((item.get("grasp") or {}).get("clock_candidates") or []))
        if bool(candidate.get("full_evaluated"))
    ]
    timing_detail = {
        "scene": dict(scene_timing),
        "pairs": _aggregate_timing_rows(results),
        "full_candidates": _aggregate_timing_rows(full_candidate_rows),
    }
    return {
        "rings_detected": len(rings),
        "mouths_detected": len(mouths),
        "matched_pairs": len(matches),
        "global_matched_pairs": len(all_matches),
        "candidate_scope": {
            "enabled": bool(candidate_scope_enabled),
            "allowed_ring_instance_ids": (
                sorted(int(value) for value in allowed_ring_ids)
                if allowed_ring_ids is not None else None
            ),
            "active_matched_ring_ids": [
                int(ring.instance_id) for ring, _mouth, _metrics in matches
            ],
        },
        "unmatched_ring_ids": [int(item.instance_id) for item in unmatched_rings],
        "unmatched_mouth_ids": [int(item.instance_id) for item in unmatched_mouths],
        "association_debug": association_debug,
        "eligible_count": len(eligible),
        "selected_ring_instance_id": selected_ring_id,
        "selected_clock_hour": selected_clock,
        "selected_clock_angle_deg_cw_from_12": selected_clock_angle,
        "selected_clock_search_batch": selected_clock_search_batch,
        "selection_scope": (
            "M36.4.2_first_valid_target_early_exit_adaptive_8_plus_4_clock_search"
            if first_valid else
            (
                "M36.4.1_staged_light_ranking_budgeted_complete_collision"
                if staged else
                "M35.2_12_clock_rim_pinch_complete_pregrasp_motion_no_post_grasp_lift"
            )
        ),
        "geometry_optimization": optimization_summary,
        "timing_ms": scene_timing,
        "timing_detail": timing_detail,
        "box_wall_model": box_wall_model,
        "neighbor_3d_model": {
            "enabled": bool(config.section("neighbor_3d").get("enabled", True)),
            "source": "aligned_depth_points_inside_other_foam_ring_instance_masks",
            "neighbor_2d_overlap_mode": str(config.section("candidate").get("neighbor_2d_overlap_mode") or "warning_only"),
            "minimum_clearance_mm": _safe_float(config.section("neighbor_3d").get("minimum_clearance_mm"), 3.0),
            "minimum_collision_points": _safe_int(config.section("neighbor_3d").get("minimum_collision_points"), 4),
        },
        "full_gripper_static_model": {
            "enabled": bool(config.section("full_gripper_static_collision").get("enabled", True)),
            "scope": "final_static_pose_only",
            "geometry_source": "measured_M35_1_dimensions",
            "components": [
                "inner_contact_block",
                "outer_contact_block",
                "inner_moving_finger",
                "outer_moving_finger",
                "palm",
                "mounting_disk",
                "pneumatic_fitting",
                "robot_wrist",
            ],
            "dynamic_sweeps_checked": False,
        },
        "full_gripper_motion_model": {
            "enabled": bool(config.section("full_gripper_motion_collision").get("enabled", True)),
            "scope": "pregrasp_to_grasp_only",
            "stages": [
                "travel_small_opening",
                "preopen_near_target",
                "approach_open",
                "insert_open",
                "close_on_rim",
            ],
            "post_grasp_lift_checked": False,
            "target_transport_checked": False,
            "geometry_source": "measured_M35_1_dimensions_with_symmetric_pivot_arc",
        },
        "robot_candidate": selected_robot_candidate,
        "instances": results,
    }
