#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB post-processing for the detergent bottle grasp task.

The model detects four semantic targets:

* large detergent bottle;
* small detergent bottle;
* bottle grasp point (``head`` in the current model);
* destination carton.

The robot-facing output keeps the existing VisionOps item field names.  A bottle
item uses ``center_px`` as the actual grasp point, while ``object_center_px``
retains the OBB centre.  Bottle ``angle_deg`` is a directed 360-degree handle
orientation resolved from the grasp point.  Carton items use their OBB centre
for ``center_px``.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

Point = Tuple[float, float]
Polygon = List[Point]


class DetectionFormatError(ValueError):
    """Raised when Runtime output cannot satisfy the configured OBB contract."""


@dataclass(frozen=True)
class DetergentGraspResult:
    image_width: int
    image_height: int
    items: List[Dict[str, Any]]
    bottles: List[Dict[str, Any]]
    boxes: List[Dict[str, Any]]
    grasp_points: List[Dict[str, Any]]
    unmatched_bottles: List[Dict[str, Any]]
    unmatched_grasp_points: List[Dict[str, Any]]
    ignored: List[Dict[str, Any]]


def _number(value: object, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return output if math.isfinite(output) else default


def _optional_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _int_set(value: object) -> set[int]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    output: set[int] = set()
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None:
            output.add(parsed)
    return output


def _name_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected OBB long-axis angle to [-90, 90]."""

    value = float(angle_deg) % 180.0
    if value > 90.0:
        value -= 180.0
    return 90.0 if abs(value + 90.0) < 1e-6 else value


def normalize_direction_angle(angle_deg: float) -> float:
    """Normalize a directed image-plane angle to [0, 360)."""

    value = float(angle_deg) % 360.0
    return 0.0 if abs(value) < 1e-9 or abs(value - 360.0) < 1e-9 else value


def resolve_handle_direction_angle(
    axis_angle_deg: float,
    object_center: Sequence[float],
    grasp_center: Sequence[float],
    long_axis_length: float,
) -> float:
    """Resolve the bottle's 360-degree handle direction from the grasp point.

    ``axis_angle_deg`` is the undirected OBB long-axis angle.  The grasp-point
    detector marks the side opposite the detergent-bottle handle, so the
    handle is defined as the long-axis endpoint farther from the grasp-point
    centre.

    Image coordinates are used: +x points right and +y points down.  Therefore
    0 degrees points right, +90 degrees points down, -90 degrees points up, and
    +/-180 degrees points left.
    """

    axis_angle = normalize_axis_angle(axis_angle_deg)
    radians = math.radians(axis_angle)
    unit_x = math.cos(radians)
    unit_y = math.sin(radians)
    center_x, center_y = _number(object_center[0]), _number(object_center[1])
    grasp_x, grasp_y = _number(grasp_center[0]), _number(grasp_center[1])
    half_length = max(0.5, abs(float(long_axis_length)) * 0.5)

    positive_end = (center_x + unit_x * half_length, center_y + unit_y * half_length)
    negative_end = (center_x - unit_x * half_length, center_y - unit_y * half_length)
    positive_distance_sq = (grasp_x - positive_end[0]) ** 2 + (grasp_y - positive_end[1]) ** 2
    negative_distance_sq = (grasp_x - negative_end[0]) ** 2 + (grasp_y - negative_end[1]) ** 2

    # The endpoint farther from the grasp point is the handle direction.  When
    # both distances are exactly equal, retain the original OBB axis direction
    # as a deterministic fallback; such a centred grasp point cannot resolve
    # the 180-degree ambiguity physically.
    directed_angle = axis_angle if positive_distance_sq >= negative_distance_sq else axis_angle + 180.0
    return normalize_direction_angle(directed_angle)


def _polygon_center(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    return (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )


def _polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += current[0] * following[1] - following[0] * current[1]
    return abs(total) * 0.5


def _scale_polygon(points: Sequence[Point], ratio: float) -> Polygon:
    center = _polygon_center(points)
    scale = max(0.1, float(ratio))
    return [
        (
            center[0] + (point[0] - center[0]) * scale,
            center[1] + (point[1] - center[1]) * scale,
        )
        for point in points
    ]


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def _rect_from_detection(detection: Mapping[str, Any]) -> Optional[Tuple[Point, float, float, float, Polygon]]:
    """Return centre, width, height, long-axis angle and polygon."""

    obb = detection.get("obb") if isinstance(detection.get("obb"), Mapping) else {}
    raw_points = obb.get("points") if isinstance(obb, Mapping) else None
    points: Polygon = []
    if isinstance(raw_points, list) and len(raw_points) >= 4:
        for raw in raw_points[:4]:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                points = []
                break
            x = _number(raw[0], float("nan"))
            y = _number(raw[1], float("nan"))
            if not (math.isfinite(x) and math.isfinite(y)):
                points = []
                break
            points.append((x, y))

    if not points and isinstance(obb, Mapping):
        cx = _number(obb.get("cx"), float("nan"))
        cy = _number(obb.get("cy"), float("nan"))
        width = _number(obb.get("w"), _number(obb.get("width"), 0.0))
        height = _number(obb.get("h"), _number(obb.get("height"), 0.0))
        if all(math.isfinite(item) for item in (cx, cy)) and width > 0.0 and height > 0.0:
            angle = _number(obb.get("angle_deg"), _number(obb.get("angle"), 0.0))
            if "angle_rad" in obb:
                angle = math.degrees(_number(obb.get("angle_rad"), 0.0))
            box = cv2.boxPoints(((float(cx), float(cy)), (float(width), float(height)), float(angle)))
            points = [(float(item[0]), float(item[1])) for item in box]

    if len(points) != 4 or _polygon_area(points) <= 1.0:
        return None

    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    (cx, cy), (width, height), angle = rect
    if width <= 0.0 or height <= 0.0:
        return None
    long_axis_angle = float(angle) if width >= height else float(angle) + 90.0
    ordered_box = cv2.boxPoints(rect)
    polygon = [(float(item[0]), float(item[1])) for item in ordered_box]
    return (float(cx), float(cy)), float(width), float(height), normalize_axis_angle(long_axis_angle), polygon


class DetergentGraspAlgorithm:
    def __init__(self, settings: Mapping[str, Any]) -> None:
        image = settings.get("image") if isinstance(settings.get("image"), Mapping) else {}
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        association = settings.get("association") if isinstance(settings.get("association"), Mapping) else {}
        selection = settings.get("selection") if isinstance(settings.get("selection"), Mapping) else {}
        output = settings.get("output") if isinstance(settings.get("output"), Mapping) else {}

        self.expected_width = max(1, int(image.get("width", 640)))
        self.expected_height = max(1, int(image.get("height", 480)))
        self.require_fixed_size = bool(image.get("require_fixed_size", True))
        self.require_obb = bool(settings.get("require_obb", True))

        self.class_ids = {
            "big_bottle": _int_set(classes.get("big_bottle_ids")),
            "small_bottle": _int_set(classes.get("small_bottle_ids")),
            "grasp_point": _int_set(classes.get("grasp_point_ids")),
            "box": _int_set(classes.get("box_ids")),
        }
        self.class_names = {
            "big_bottle": _name_set(classes.get("big_bottle_names")),
            "small_bottle": _name_set(classes.get("small_bottle_names")),
            "grasp_point": _name_set(classes.get("grasp_point_names")),
            "box": _name_set(classes.get("box_names")),
        }
        self.thresholds = {
            "big_bottle": float(classes.get("big_bottle_min_confidence", 0.5)),
            "small_bottle": float(classes.get("small_bottle_min_confidence", 0.5)),
            "grasp_point": float(classes.get("grasp_point_min_confidence", 0.5)),
            "box": float(classes.get("box_min_confidence", 0.5)),
        }

        self.expand_ratio = max(1.0, float(association.get("bottle_polygon_expand_ratio", 1.18)))
        self.max_center_distance_ratio = max(0.0, float(association.get("max_center_distance_ratio", 0.65)))
        self.require_grasp_point = bool(association.get("require_grasp_point", True))
        self.max_bottles = max(1, int(selection.get("max_bottles", 8)))
        self.max_boxes = max(1, int(selection.get("max_boxes", 1)))
        self.output_order = str(selection.get("output_order", "row_major")).strip().lower()
        if self.output_order not in {"row_major", "column_major", "confidence"}:
            raise ValueError("selection.output_order 必须为 row_major/column_major/confidence")
        self.include_obb_points = bool(output.get("include_obb_points", True))
        self.include_class_name = bool(output.get("include_class_name", True))

    @staticmethod
    def _image_size(runtime_result: Mapping[str, Any]) -> Tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        width = int(_number(image.get("width")))
        height = int(_number(image.get("height")))
        if width <= 0 or height <= 0:
            raise DetectionFormatError("Runtime inference_result 缺少有效 image.width/image.height")
        return width, height

    def _semantic(self, class_id: Optional[int], class_name: str) -> Optional[str]:
        lower = class_name.strip().lower()
        # Class names take precedence, so a newly trained model can reorder IDs
        # without silently changing semantics when labels remain stable.
        if lower:
            for semantic, names in self.class_names.items():
                if lower in names:
                    return semantic
        if class_id is not None:
            for semantic, ids in self.class_ids.items():
                if class_id in ids:
                    return semantic
        return None

    def _parse_detection(self, raw: Mapping[str, Any], index: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        class_id = _optional_int(raw.get("class_id"))
        class_name = str(raw.get("class_name") or "")
        semantic = self._semantic(class_id, class_name)
        source_id = str(raw.get("id") or "det-{}".format(index))
        if semantic is None:
            return None, {
                "source_id": source_id,
                "reason": "class_not_used",
                "class_id": class_id,
                "class_name": class_name,
            }
        score = _number(raw.get("score"), _number(raw.get("confidence"), 0.0))
        if score < self.thresholds[semantic]:
            return None, {
                "source_id": source_id,
                "reason": "low_confidence",
                "semantic": semantic,
                "score": score,
            }
        rect = _rect_from_detection(raw)
        if rect is None:
            if self.require_obb:
                return None, {
                    "source_id": source_id,
                    "reason": "missing_or_invalid_obb",
                    "semantic": semantic,
                }
            bbox = raw.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None, {"source_id": source_id, "reason": "missing_geometry", "semantic": semantic}
            x1, y1, x2, y2 = [_number(value) for value in bbox[:4]]
            if x2 <= x1 or y2 <= y1:
                return None, {"source_id": source_id, "reason": "invalid_bbox", "semantic": semantic}
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rect = ((x1 + x2) * 0.5, (y1 + y2) * 0.5), x2 - x1, y2 - y1, 0.0, polygon

        center, width, height, angle_deg, polygon = rect
        parsed = {
            "source_id": source_id,
            "semantic": semantic,
            "class_id": class_id if class_id is not None else -1,
            "class_name": class_name or semantic,
            "confidence": max(0.0, min(1.0, score)),
            "center": [float(center[0]), float(center[1])],
            "width": width,
            "height": height,
            "angle_deg": angle_deg,
            "polygon": [[float(point[0]), float(point[1])] for point in polygon],
            "raw": deepcopy(dict(raw)),
        }
        return parsed, None

    def _sort_targets(self, values: List[Dict[str, Any]]) -> None:
        if self.output_order == "row_major":
            values.sort(key=lambda item: (float(item["center"][1]), float(item["center"][0])))
        elif self.output_order == "column_major":
            values.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
        else:
            values.sort(key=lambda item: -float(item["confidence"]))

    def _associate(
        self,
        bottles: Sequence[Mapping[str, Any]],
        grasp_points: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates: List[Tuple[int, float, float, int, int]] = []
        for bottle_index, bottle in enumerate(bottles):
            polygon = [(float(item[0]), float(item[1])) for item in bottle["polygon"]]
            expanded = _scale_polygon(polygon, self.expand_ratio)
            cx, cy = float(bottle["center"][0]), float(bottle["center"][1])
            diagonal = max(1.0, math.hypot(float(bottle["width"]), float(bottle["height"])))
            for grasp_index, grasp in enumerate(grasp_points):
                gx, gy = float(grasp["center"][0]), float(grasp["center"][1])
                distance_ratio = math.hypot(gx - cx, gy - cy) / diagonal
                inside = _point_in_polygon((gx, gy), expanded)
                if not inside and distance_ratio > self.max_center_distance_ratio:
                    continue
                candidates.append(
                    (
                        0 if inside else 1,
                        distance_ratio,
                        -float(grasp["confidence"]),
                        bottle_index,
                        grasp_index,
                    )
                )

        candidates.sort()
        assigned_bottles: set[int] = set()
        assigned_grasps: set[int] = set()
        matches: Dict[str, Dict[str, Any]] = {}
        for _outside_rank, _distance, _negative_score, bottle_index, grasp_index in candidates:
            if bottle_index in assigned_bottles or grasp_index in assigned_grasps:
                continue
            assigned_bottles.add(bottle_index)
            assigned_grasps.add(grasp_index)
            bottle = bottles[bottle_index]
            matches[str(bottle["source_id"])] = deepcopy(dict(grasp_points[grasp_index]))

        unmatched_bottles = [deepcopy(dict(value)) for index, value in enumerate(bottles) if index not in assigned_bottles]
        unmatched_grasps = [deepcopy(dict(value)) for index, value in enumerate(grasp_points) if index not in assigned_grasps]
        return matches, unmatched_bottles, unmatched_grasps

    def _bottle_item(self, item_id: int, bottle: Mapping[str, Any], grasp: Mapping[str, Any]) -> Dict[str, Any]:
        grasp_center = [round(_number(value), 3) for value in list(grasp["center"])[:2]]
        object_center = [round(_number(value), 3) for value in list(bottle["center"])[:2]]
        document: Dict[str, Any] = {
            "id": int(item_id),
            "class_id": int(bottle["class_id"]),
            "confidence": round(float(bottle["confidence"]), 6),
            "position_camera": [0.0, 0.0, 0.0],
            "angle_deg": round(
                resolve_handle_direction_angle(
                    float(bottle["angle_deg"]),
                    bottle["center"],
                    grasp["center"],
                    max(float(bottle["width"]), float(bottle["height"])),
                ),
                3,
            ),
            "center_px": grasp_center,
            "type": None,
            "target_type": str(bottle["semantic"]),
            "object_center_px": object_center,
            "grasp_point_px": grasp_center,
            "grasp_confidence": round(float(grasp["confidence"]), 6),
            "source_detection_id": str(bottle["source_id"]),
            "grasp_source_detection_id": str(grasp["source_id"]),
        }
        if self.include_class_name:
            document["class_name"] = str(bottle["class_name"])
        if self.include_obb_points:
            document["obb_points"] = [
                [round(_number(point[0]), 3), round(_number(point[1]), 3)] for point in bottle["polygon"]
            ]
        return document

    def _box_item(self, item_id: int, box: Mapping[str, Any]) -> Dict[str, Any]:
        center = [round(_number(value), 3) for value in list(box["center"])[:2]]
        document: Dict[str, Any] = {
            "id": int(item_id),
            "class_id": int(box["class_id"]),
            "confidence": round(float(box["confidence"]), 6),
            "position_camera": [0.0, 0.0, 0.0],
            "angle_deg": round(normalize_axis_angle(float(box["angle_deg"])), 3),
            "center_px": center,
            "type": None,
            "target_type": "box",
            "object_center_px": center,
            "grasp_point_px": None,
            "source_detection_id": str(box["source_id"]),
        }
        if self.include_class_name:
            document["class_name"] = str(box["class_name"])
        if self.include_obb_points:
            document["obb_points"] = [
                [round(_number(point[0]), 3), round(_number(point[1]), 3)] for point in box["polygon"]
            ]
        return document

    def evaluate(self, runtime_result: Mapping[str, Any]) -> DetergentGraspResult:
        width, height = self._image_size(runtime_result)
        if self.require_fixed_size and (width != self.expected_width or height != self.expected_height):
            raise DetectionFormatError(
                "detergent_grasp 固定图像尺寸为 {}x{}，Runtime 当前为 {}x{}".format(
                    self.expected_width, self.expected_height, width, height
                )
            )

        detections = runtime_result.get("detections")
        detections = detections if isinstance(detections, list) else []
        parsed: List[Dict[str, Any]] = []
        ignored: List[Dict[str, Any]] = []
        for index, raw in enumerate(detections):
            if not isinstance(raw, Mapping):
                ignored.append({"source_id": "det-{}".format(index), "reason": "not_an_object"})
                continue
            item, error = self._parse_detection(raw, index)
            if item is not None:
                parsed.append(item)
            if error is not None:
                ignored.append(error)

        bottles = [item for item in parsed if item["semantic"] in {"big_bottle", "small_bottle"}]
        boxes = [item for item in parsed if item["semantic"] == "box"]
        grasp_points = [item for item in parsed if item["semantic"] == "grasp_point"]
        self._sort_targets(bottles)
        self._sort_targets(boxes)
        self._sort_targets(grasp_points)
        bottles = bottles[: self.max_bottles]
        boxes = sorted(boxes, key=lambda item: -float(item["confidence"]))[: self.max_boxes]

        matches, unmatched_bottles, unmatched_grasp_points = self._associate(bottles, grasp_points)
        output: List[Dict[str, Any]] = []
        next_id = 0
        for bottle in bottles:
            grasp = matches.get(str(bottle["source_id"]))
            if grasp is None and self.require_grasp_point:
                continue
            if grasp is None:
                grasp = {
                    "center": deepcopy(bottle["center"]),
                    "confidence": 0.0,
                    "source_id": "fallback-object-center",
                }
            output.append(self._bottle_item(next_id, bottle, grasp))
            next_id += 1
        for box in boxes:
            output.append(self._box_item(next_id, box))
            next_id += 1

        return DetergentGraspResult(
            image_width=width,
            image_height=height,
            items=output,
            bottles=[deepcopy(item) for item in bottles],
            boxes=[deepcopy(item) for item in boxes],
            grasp_points=[deepcopy(item) for item in grasp_points],
            unmatched_bottles=unmatched_bottles,
            unmatched_grasp_points=unmatched_grasp_points,
            ignored=ignored,
        )
