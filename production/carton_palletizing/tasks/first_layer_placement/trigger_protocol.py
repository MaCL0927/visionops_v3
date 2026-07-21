#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-mode robot protocol helpers for carton palletizing.

The robot-facing contract follows ``external_box_protocol`` v2.0-draft:
``trigger`` requests are correlated by ``trigger_task_id`` and every detection
item contains an OBB-derived point/angle in both pixel and camera coordinates.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


Point = Tuple[float, float]
Polygon = List[Point]

FAULT_NONE = 0
FAULT_CAMERA_DISCONNECTED = 3101
FAULT_VISION_INFERENCE_ERROR = 3201
FAULT_TYPE_NONE = "NONE"
FAULT_TYPE_CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
FAULT_TYPE_VISION_INFERENCE_ERROR = "VISION_INFERENCE_ERROR"


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def point(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = number(value[0], float("nan"))
    y = number(value[1], float("nan"))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def polygon_center(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    return (
        sum(item[0] for item in points) / float(len(points)),
        sum(item[1] for item in points) / float(len(points)),
    )


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    value = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        value += current[0] * following[1] - following[0] * current[1]
    return abs(value) / 2.0


def point_in_polygon(value: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    x, y = value
    previous = polygon[-1]
    for current in polygon:
        if ((current[1] > y) != (previous[1] > y)) and (
            x
            < (previous[0] - current[0])
            * (y - current[1])
            / ((previous[1] - current[1]) or 1e-9)
            + current[0]
        ):
            inside = not inside
        previous = current
    return inside


def scale_polygon(points: Sequence[Point], scale: float) -> Polygon:
    center = polygon_center(points)
    safe_scale = max(0.1, float(scale))
    return [
        (
            center[0] + (item[0] - center[0]) * safe_scale,
            center[1] + (item[1] - center[1]) * safe_scale,
        )
        for item in points
    ]


def polygon_overlap_ratio(subject: Sequence[Point], reference: Sequence[Point]) -> float:
    """Return intersection area divided by subject area for convex OBB quads."""

    subject_area = polygon_area(subject)
    if subject_area <= 1e-6 or len(subject) < 3 or len(reference) < 3:
        return 0.0
    left = cv2.convexHull(np.asarray(subject, dtype=np.float32)).reshape(-1, 2)
    right = cv2.convexHull(np.asarray(reference, dtype=np.float32)).reshape(-1, 2)
    try:
        intersection, _ = cv2.intersectConvexConvex(left, right)
    except cv2.error:
        return 0.0
    return max(0.0, min(1.0, float(intersection) / subject_area))


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected OBB axis angle to [-90, 90]."""

    value = float(angle_deg) % 180.0
    if value > 90.0:
        value -= 180.0
    return 90.0 if abs(value + 90.0) < 1e-6 else value


def sample_depth_mm(
    depth_image: "np.ndarray",
    center_px: Sequence[float],
    image_width: int,
    image_height: int,
    radius_px: int,
    percentile: float,
    min_valid_pixels: int,
    min_depth_mm: int,
    max_depth_mm: int,
) -> Dict[str, Any]:
    """Sample a robust aligned depth value around one RGB-space point."""

    parsed = point(center_px)
    if parsed is None or depth_image.ndim != 2:
        return {"valid": False, "depth_mm": 0.0, "sample_px": [0, 0], "valid_pixels": 0}
    depth_height, depth_width = depth_image.shape[:2]
    if image_width <= 0 or image_height <= 0 or depth_width <= 0 or depth_height <= 0:
        return {"valid": False, "depth_mm": 0.0, "sample_px": [0, 0], "valid_pixels": 0}

    dx = int(round(parsed[0] * float(depth_width) / float(image_width)))
    dy = int(round(parsed[1] * float(depth_height) / float(image_height)))
    dx = min(max(dx, 0), depth_width - 1)
    dy = min(max(dy, 0), depth_height - 1)
    radius = max(0, int(radius_px))
    x1, x2 = max(0, dx - radius), min(depth_width, dx + radius + 1)
    y1, y2 = max(0, dy - radius), min(depth_height, dy + radius + 1)
    roi = depth_image[y1:y2, x1:x2]
    valid = roi[(roi >= int(min_depth_mm)) & (roi <= int(max_depth_mm))]
    if valid.size < max(1, int(min_valid_pixels)):
        return {
            "valid": False,
            "depth_mm": 0.0,
            "sample_px": [dx, dy],
            "valid_pixels": int(valid.size),
        }
    depth = float(np.percentile(valid.astype(np.float32), float(percentile)))
    return {
        "valid": math.isfinite(depth),
        "depth_mm": round(depth, 3) if math.isfinite(depth) else 0.0,
        "sample_px": [dx, dy],
        "valid_pixels": int(valid.size),
    }


def candidate_depths(
    boxes: Sequence[Mapping[str, Any]],
    depth_image: "np.ndarray",
    image_width: int,
    image_height: int,
    settings: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    output = []  # type: List[Dict[str, Any]]
    for index, raw in enumerate(boxes):
        item = deepcopy(dict(raw))
        center = item.get("center")
        depth = sample_depth_mm(
            depth_image,
            center if isinstance(center, (list, tuple)) else [0.0, 0.0],
            image_width,
            image_height,
            int(settings.get("roi_radius_px", 6)),
            float(settings.get("percentile", 50.0)),
            int(settings.get("min_valid_pixels", 3)),
            int(settings.get("min_depth_mm", 100)),
            int(settings.get("max_depth_mm", 5000)),
        )
        item["depth"] = depth
        item["candidate_index"] = index
        output.append(item)
    return output


def select_held_box(
    boxes: Sequence[Mapping[str, Any]],
    tray_polygon: Optional[Sequence[Point]],
    depth_image: "np.ndarray",
    image_width: int,
    image_height: int,
    settings: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select the robot-held box without changing code between deployment stages.

    ``nearest_depth`` is intended for the initial stationary-robot/two-layer
    phase. ``outside_tray`` is intended for the later mobile-robot phase.
    """

    mode = str(settings.get("mode") or "nearest_depth").strip().lower()
    depth_settings = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}
    candidates = candidate_depths(boxes, depth_image, image_width, image_height, depth_settings)
    diagnostics = {
        "mode": mode,
        "candidate_count": len(candidates),
        "candidates": [],
        "reason": "NO_BOX",
    }  # type: Dict[str, Any]
    if not candidates:
        return None, diagnostics

    for item in candidates:
        diagnostics["candidates"].append(
            {
                "source_id": str(item.get("id") or ""),
                "score": round(number(item.get("score")), 6),
                "center_px": [round(number(value), 3) for value in item.get("center", [0.0, 0.0])[:2]],
                "depth": deepcopy(item.get("depth")),
            }
        )

    if mode == "nearest_depth":
        valid = [item for item in candidates if bool(item.get("depth", {}).get("valid"))]
        if not valid:
            diagnostics["reason"] = "NO_VALID_DEPTH"
            return None, diagnostics
        valid.sort(key=lambda item: (number(item["depth"].get("depth_mm"), 1e12), -number(item.get("score"))))
        selected = valid[0]
        advantage = None
        if len(valid) > 1:
            advantage = number(valid[1]["depth"].get("depth_mm")) - number(selected["depth"].get("depth_mm"))
            min_advantage = float(settings.get("min_depth_advantage_mm", 20.0))
            if bool(settings.get("require_depth_advantage", False)) and advantage < min_advantage:
                diagnostics["reason"] = "DEPTH_AMBIGUOUS"
                diagnostics["depth_advantage_mm"] = round(advantage, 3)
                return None, diagnostics
        diagnostics["reason"] = "SELECTED_NEAREST_DEPTH"
        diagnostics["depth_advantage_mm"] = round(advantage, 3) if advantage is not None else None
        diagnostics["selected_source_id"] = str(selected.get("id") or "")
        return selected, diagnostics

    if mode == "outside_tray":
        if not tray_polygon or len(tray_polygon) < 3:
            diagnostics["reason"] = "TRAY_REFERENCE_UNAVAILABLE"
            return None, diagnostics
        outside = []  # type: List[Dict[str, Any]]
        expanded = scale_polygon(tray_polygon, 1.0 + max(0.0, float(settings.get("tray_expand_ratio", 0.05))))
        max_overlap = float(settings.get("max_tray_overlap_ratio", 0.20))
        for item in candidates:
            center = point(item.get("center")) or (0.0, 0.0)
            polygon = item.get("polygon") if isinstance(item.get("polygon"), list) else []
            overlap = polygon_overlap_ratio(polygon, tray_polygon)
            center_inside = point_in_polygon(center, expanded)
            item["tray_overlap_ratio"] = overlap
            item["center_inside_expanded_tray"] = center_inside
            if not center_inside and overlap <= max_overlap:
                outside.append(item)
        for detail, item in zip(diagnostics["candidates"], candidates):
            detail["tray_overlap_ratio"] = round(number(item.get("tray_overlap_ratio")), 6)
            detail["center_inside_expanded_tray"] = bool(item.get("center_inside_expanded_tray"))
        if not outside:
            diagnostics["reason"] = "NO_OUTSIDE_TRAY_BOX"
            return None, diagnostics
        prefer_nearest = bool(settings.get("outside_tray_prefer_nearest_depth", True))
        if prefer_nearest:
            outside.sort(
                key=lambda item: (
                    0 if bool(item.get("depth", {}).get("valid")) else 1,
                    number(item.get("depth", {}).get("depth_mm"), 1e12),
                    -number(item.get("score")),
                )
            )
        else:
            outside.sort(key=lambda item: (-number(item.get("score")), number(item.get("tray_overlap_ratio"))))
        selected = outside[0]
        diagnostics["reason"] = "SELECTED_OUTSIDE_TRAY"
        diagnostics["selected_source_id"] = str(selected.get("id") or "")
        return selected, diagnostics

    raise ValueError("held_box_selection.mode 必须为 nearest_depth 或 outside_tray")



def select_top_surface_targets(
    boxes: Sequence[Mapping[str, Any]],
    trays: Sequence[Mapping[str, Any]],
    tray_polygon: Optional[Sequence[Point]],
    depth_image: "np.ndarray",
    image_width: int,
    image_height: int,
    settings: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Select the visible top support surface for the robot placement task.

    The M29 robot contract no longer asks VisionOps to plan a slot or infer a
    layer number.  VisionOps only returns the currently visible support
    objects:

    * no carton detected -> the best detected tray (one target);
    * carton(s) detected -> only cartons belonging to the nearest/top depth
      cluster (one to ``max_items`` targets, normally at most four).

    Depth is sampled at each OBB centre.  Candidates are sorted from near to
    far and the first cluster is cut when either the consecutive depth gap or
    the total span from the nearest candidate exceeds the configured limits.
    This avoids hard-coding a box height while still separating visible layers.
    """

    depth_settings = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}
    max_items = max(1, int(settings.get("max_items", 4)))
    layer_gap_mm = max(1.0, float(settings.get("layer_gap_mm", 80.0)))
    max_top_span_mm = max(layer_gap_mm, float(settings.get("max_top_layer_span_mm", 140.0)))
    filter_to_tray = bool(settings.get("filter_to_tray_region", True))
    tray_expand_ratio = max(0.0, float(settings.get("tray_expand_ratio", 0.08)))
    min_overlap = max(0.0, min(1.0, float(settings.get("min_tray_overlap_ratio", 0.05))))
    allow_without_tray = bool(settings.get("allow_boxes_without_tray_reference", True))
    sort_order = str(settings.get("sort_order") or "image_yx").strip().lower()

    diagnostics = {
        "candidate_box_count": len(boxes),
        "candidate_tray_count": len(trays),
        "filter_to_tray_region": filter_to_tray,
        "layer_gap_mm": layer_gap_mm,
        "max_top_layer_span_mm": max_top_span_mm,
        "max_items": max_items,
        "reason": "NO_TARGET",
        "boxes": [],
    }  # type: Dict[str, Any]

    expanded_tray = None  # type: Optional[Polygon]
    if tray_polygon and len(tray_polygon) >= 3:
        expanded_tray = scale_polygon(
            [(number(value[0]), number(value[1])) for value in tray_polygon],
            1.0 + tray_expand_ratio,
        )

    scoped_boxes = []  # type: List[Mapping[str, Any]]
    for raw in boxes:
        center = point(raw.get("center"))
        polygon = raw.get("polygon") if isinstance(raw.get("polygon"), list) else []
        overlap = polygon_overlap_ratio(polygon, tray_polygon or [])
        center_inside = bool(center is not None and expanded_tray and point_in_polygon(center, expanded_tray))
        accepted = True
        if filter_to_tray:
            if expanded_tray is None:
                accepted = allow_without_tray
            else:
                accepted = center_inside or overlap >= min_overlap
        diagnostics["boxes"].append(
            {
                "source_id": str(raw.get("id") or ""),
                "score": round(number(raw.get("score")), 6),
                "center_px": [round(number(value), 3) for value in (raw.get("center") or [0.0, 0.0])[:2]],
                "tray_overlap_ratio": round(overlap, 6),
                "center_inside_expanded_tray": center_inside,
                "accepted_by_region": accepted,
            }
        )
        if accepted:
            scoped_boxes.append(raw)

    if scoped_boxes:
        candidates = candidate_depths(
            scoped_boxes,
            depth_image,
            image_width,
            image_height,
            depth_settings,
        )
        valid = [item for item in candidates if bool(item.get("depth", {}).get("valid"))]
        depth_by_source = {
            str(item.get("id") or ""): deepcopy(item.get("depth")) for item in candidates
        }
        for detail in diagnostics["boxes"]:
            if detail["accepted_by_region"]:
                detail["depth"] = depth_by_source.get(detail["source_id"])
        if not valid:
            diagnostics["reason"] = "BOX_DEPTH_UNAVAILABLE"
            return [], "box", diagnostics

        valid.sort(
            key=lambda item: (
                number(item.get("depth", {}).get("depth_mm"), 1e12),
                -number(item.get("score")),
            )
        )
        nearest_depth = number(valid[0].get("depth", {}).get("depth_mm"), 1e12)
        previous_depth = nearest_depth
        top_cluster = []  # type: List[Dict[str, Any]]
        for item in valid:
            current_depth = number(item.get("depth", {}).get("depth_mm"), 1e12)
            if top_cluster and (
                current_depth - previous_depth > layer_gap_mm
                or current_depth - nearest_depth > max_top_span_mm
            ):
                break
            top_cluster.append(item)
            previous_depth = current_depth

        # More than four OBBs usually means duplicates or an object outside the
        # actual stack.  Keep the strongest four, then restore deterministic
        # image order for the robot consumer.
        if len(top_cluster) > max_items:
            top_cluster = sorted(
                top_cluster,
                key=lambda item: (-number(item.get("score")), number(item.get("depth", {}).get("depth_mm"))),
            )[:max_items]

        if sort_order == "image_xy":
            top_cluster.sort(key=lambda item: (number((item.get("center") or [0, 0])[0]), number((item.get("center") or [0, 0])[1])))
        elif sort_order == "confidence":
            top_cluster.sort(key=lambda item: -number(item.get("score")))
        else:  # image_yx
            top_cluster.sort(key=lambda item: (number((item.get("center") or [0, 0])[1]), number((item.get("center") or [0, 0])[0])))

        diagnostics["reason"] = "TOP_LAYER_BOXES_SELECTED"
        diagnostics["nearest_depth_mm"] = round(nearest_depth, 3)
        diagnostics["selected_count"] = len(top_cluster)
        diagnostics["selected_source_ids"] = [str(item.get("id") or "") for item in top_cluster]
        diagnostics["selected_depths_mm"] = [
            round(number(item.get("depth", {}).get("depth_mm")), 3) for item in top_cluster
        ]
        return top_cluster, "box", diagnostics

    if boxes and filter_to_tray:
        diagnostics["excluded_box_count"] = len(boxes)

    if not trays:
        diagnostics["reason"] = "NO_TRAY_OR_BOX"
        return [], "none", diagnostics

    selected_tray = max(
        trays,
        key=lambda item: (number(item.get("score")), polygon_area(item.get("polygon") or [])),
    )
    tray_with_depth = candidate_depths(
        [selected_tray],
        depth_image,
        image_width,
        image_height,
        depth_settings,
    )[0]
    if not bool(tray_with_depth.get("depth", {}).get("valid")):
        diagnostics["reason"] = "TRAY_DEPTH_UNAVAILABLE"
        diagnostics["tray_depth"] = deepcopy(tray_with_depth.get("depth"))
        return [], "tray", diagnostics

    diagnostics["reason"] = "TRAY_SELECTED"
    diagnostics["selected_count"] = 1
    diagnostics["selected_source_ids"] = [str(tray_with_depth.get("id") or "")]
    diagnostics["selected_depths_mm"] = [
        round(number(tray_with_depth.get("depth", {}).get("depth_mm")), 3)
    ]
    return [tray_with_depth], "tray", diagnostics

def protocol_item(
    item_id: int,
    class_id: int,
    confidence: float,
    center_px: Sequence[float],
    position_camera: Sequence[float],
    angle_deg: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    document = {
        "id": int(item_id),
        "class_id": int(class_id),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "position_camera": [round(number(value), 3) for value in list(position_camera)[:3]],
        "angle_deg": round(normalize_axis_angle(angle_deg), 3),
        "center_px": [round(number(value), 3) for value in list(center_px)[:2]],
        "type": None,
    }
    if extra:
        document.update(deepcopy(dict(extra)))
    return document
