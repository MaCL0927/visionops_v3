"""OBB-based first-layer carton placement planning.

Phase 1 is RGB-only. The application locks the tray OBB, projects four
configured slots into the tray quadrilateral, matches carton OBBs to those
slots, and emits overlay-ready placement state.

The module intentionally supports Python 3.8, which is the system Python on
LB3576 images used by this project.
"""

from __future__ import annotations

import math
import threading
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Polygon = List[Point]


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _point(value: object) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0], float("nan"))
    y = _number(value[1], float("nan"))
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _bbox(value: object) -> Optional[BBox]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x1, y1, x2, y2 = (_number(item) for item in value[:4])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_polygon(box: BBox) -> Polygon:
    return [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]


def _bbox_center(box: BBox) -> Point:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(left: BBox, right: BBox) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _polygon_bbox(points: Sequence[Point]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_center(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    return (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )


def _signed_polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += current[0] * following[1] - following[0] * current[1]
    return total * 0.5


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(_signed_polygon_area(points))


def _cross(left: Point, right: Point) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(left: Point, right: Point) -> Point:
    return left[0] - right[0], left[1] - right[1]


def _line_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point:
    first = _subtract(p2, p1)
    second = _subtract(q2, q1)
    denominator = _cross(first, second)
    if abs(denominator) < 1e-9:
        return p2
    t = _cross(_subtract(q1, p1), second) / denominator
    return p1[0] + t * first[0], p1[1] + t * first[1]


def _convex_intersection(subject: Sequence[Point], clip_polygon: Sequence[Point]) -> Polygon:
    """Return the intersection polygon for two convex polygons."""

    output = list(subject)
    if len(output) < 3 or len(clip_polygon) < 3:
        return []
    orientation = 1.0 if _signed_polygon_area(clip_polygon) >= 0.0 else -1.0

    def inside(point_value: Point, edge_start: Point, edge_end: Point) -> bool:
        edge = _subtract(edge_end, edge_start)
        relative = _subtract(point_value, edge_start)
        return orientation * _cross(edge, relative) >= -1e-6

    for index, clip_start in enumerate(clip_polygon):
        clip_end = clip_polygon[(index + 1) % len(clip_polygon)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = inside(previous, clip_start, clip_end)
        for current in input_points:
            current_inside = inside(current, clip_start, clip_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
            previous_inside = current_inside
    return output


def polygon_iou(left: Sequence[Point], right: Sequence[Point]) -> float:
    left_area = _polygon_area(left)
    right_area = _polygon_area(right)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    intersection = _polygon_area(_convex_intersection(left, right))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _point_in_polygon(point_value: Point, polygon: Sequence[Point]) -> bool:
    x, y = point_value
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if ((current[1] > y) != (previous[1] > y)) and (
            x
            < (previous[0] - current[0])
            * (y - current[1])
            / ((previous[1] - current[1]) or 1e-9)
            + current[0]
        ):
            inside = not inside
        j = i
    return inside


def _order_quad(points: Sequence[Point]) -> Polygon:
    """Canonicalize four OBB points as image TL, TR, BR, BL.

    Rockchip Runtime emits the four points in OBB-local order. Canonicalizing
    them prevents slot templates from jumping by 90 degrees if the model swaps
    width/height representation between adjacent frames.
    """

    if len(points) != 4:
        return list(points)
    center = _polygon_center(points)
    ordered = sorted(points, key=lambda item: math.atan2(item[1] - center[1], item[0] - center[0]))
    start = min(range(4), key=lambda idx: (ordered[idx][0] + ordered[idx][1], ordered[idx][1], ordered[idx][0]))
    ordered = ordered[start:] + ordered[:start]
    # From the top-left point, the next point must be the right-side neighbour.
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0]] + list(reversed(ordered[1:]))
    return ordered


def _obb_polygon(detection: Mapping[str, Any]) -> Optional[Polygon]:
    obb = detection.get("obb")
    points_value = obb.get("points") if isinstance(obb, Mapping) else None
    if not isinstance(points_value, list) or len(points_value) != 4:
        return None
    points = []  # type: Polygon
    for value in points_value:
        parsed = _point(value)
        if parsed is None:
            return None
        points.append(parsed)
    ordered = _order_quad(points)
    if _polygon_area(ordered) <= 1.0:
        return None
    return ordered


def _center_from_detection(detection: Mapping[str, Any], polygon: Sequence[Point], box: BBox) -> Point:
    value = detection.get("center_xy")
    parsed = _point(value)
    if parsed is not None:
        return parsed
    obb = detection.get("obb")
    if isinstance(obb, Mapping):
        parsed = _point([obb.get("cx"), obb.get("cy")])
        if parsed is not None:
            return parsed
    if polygon:
        return _polygon_center(polygon)
    return _bbox_center(box)


def _edge_angle_deg(start: Point, end: Point) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _normalize_axis_angle(angle_deg: float) -> float:
    normalized = angle_deg % 180.0
    return normalized + 180.0 if normalized < 0.0 else normalized


def _axis_angle_diff(left_deg: float, right_deg: float) -> float:
    difference = abs(_normalize_axis_angle(left_deg) - _normalize_axis_angle(right_deg))
    return min(difference, 180.0 - difference)


def _long_axis_angle(points: Sequence[Point]) -> float:
    if len(points) < 4:
        return 0.0
    edges = []  # type: List[Tuple[float, float]]
    for index in range(4):
        start = points[index]
        end = points[(index + 1) % 4]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        edges.append((length, _edge_angle_deg(start, end)))
    return _normalize_axis_angle(max(edges, key=lambda item: item[0])[1])


def _bilinear_project(quad: Sequence[Point], u: float, v: float) -> Point:
    """Project normalized coordinates into a TL/TR/BR/BL quadrilateral."""

    top_left, top_right, bottom_right, bottom_left = quad
    x = (
        (1.0 - u) * (1.0 - v) * top_left[0]
        + u * (1.0 - v) * top_right[0]
        + u * v * bottom_right[0]
        + (1.0 - u) * v * bottom_left[0]
    )
    y = (
        (1.0 - u) * (1.0 - v) * top_left[1]
        + u * (1.0 - v) * top_right[1]
        + u * v * bottom_right[1]
        + (1.0 - u) * v * bottom_left[1]
    )
    return x, y


def _edge_length(start: Point, end: Point) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _centered_square_quad(tray_quad: Sequence[Point], fill_ratio: float = 1.0) -> Tuple[Polygon, Dict[str, float]]:
    """Build the largest centered square inside the detected tray OBB.

    The tray in this task is longer in one direction. The stacking footprint
    therefore uses the short physical edge as the square side, occupies the
    short direction completely, and leaves equal margins at both ends of the
    long direction. Coordinates remain expressed through bilinear projection
    so the footprint follows tray rotation and mild perspective distortion.
    """

    if len(tray_quad) != 4:
        return list(tray_quad), {
            "u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0,
            "tray_width_px": 0.0, "tray_height_px": 0.0, "square_side_px": 0.0,
        }

    top_left, top_right, bottom_right, bottom_left = tray_quad
    tray_width = (_edge_length(top_left, top_right) + _edge_length(bottom_left, bottom_right)) / 2.0
    tray_height = (_edge_length(top_left, bottom_left) + _edge_length(top_right, bottom_right)) / 2.0
    safe_fill = min(1.0, max(0.1, float(fill_ratio)))
    square_side = max(1.0, min(tray_width, tray_height) * safe_fill)
    u_span = min(1.0, square_side / max(tray_width, 1e-6))
    v_span = min(1.0, square_side / max(tray_height, 1e-6))
    u_min = (1.0 - u_span) / 2.0
    u_max = 1.0 - u_min
    v_min = (1.0 - v_span) / 2.0
    v_max = 1.0 - v_min
    footprint = [
        _bilinear_project(tray_quad, u_min, v_min),
        _bilinear_project(tray_quad, u_max, v_min),
        _bilinear_project(tray_quad, u_max, v_max),
        _bilinear_project(tray_quad, u_min, v_max),
    ]
    return footprint, {
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "tray_width_px": tray_width,
        "tray_height_px": tray_height,
        "square_side_px": square_side,
    }


class FirstLayerPlacementAlgorithm:
    """Stateful four-slot occupancy tracker for an OBB palletizing model."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = deepcopy(dict(settings))
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        geometry = settings.get("geometry") if isinstance(settings.get("geometry"), Mapping) else {}
        tracking = settings.get("tray_tracking") if isinstance(settings.get("tray_tracking"), Mapping) else {}
        matching = settings.get("matching") if isinstance(settings.get("matching"), Mapping) else {}
        temporal = settings.get("temporal") if isinstance(settings.get("temporal"), Mapping) else {}
        template = settings.get("template") if isinstance(settings.get("template"), Mapping) else {}

        self.tray_ids = {int(item) for item in classes.get("tray_class_ids", [])}
        self.tray_names = {str(item).strip().lower() for item in classes.get("tray_class_names", [])}
        self.box_ids = {int(item) for item in classes.get("box_class_ids", [])}
        self.box_names = {str(item).strip().lower() for item in classes.get("box_class_names", [])}
        self.tray_min_confidence = float(classes.get("tray_min_confidence", 0.5))
        self.box_min_confidence = float(classes.get("box_min_confidence", 0.5))
        self.require_obb = bool(geometry.get("require_obb", True))
        self.footprint_mode = str(geometry.get("footprint_mode", "centered_square_by_short_edge")).strip().lower()
        self.footprint_fill_ratio = float(geometry.get("footprint_fill_ratio", 1.0))

        self.lock_after_first_detection = bool(tracking.get("lock_after_first_detection", True))
        self.ema_alpha = float(tracking.get("ema_alpha", 0.35))
        self.update_min_iou = float(tracking.get("update_min_iou", 0.30))

        self.min_iou = float(matching.get("min_iou", 0.12))
        self.max_center_distance_ratio = float(matching.get("max_center_distance_ratio", 0.60))
        self.max_orientation_diff_deg = float(matching.get("max_orientation_diff_deg", 45.0))
        self.center_inside_bonus = float(matching.get("center_inside_bonus", 0.45))
        self.iou_weight = float(matching.get("iou_weight", 0.35))
        self.center_weight = float(matching.get("center_weight", 0.20))
        self.orientation_weight = float(matching.get("orientation_weight", 0.10))

        self.occupied_confirm_frames = max(1, int(temporal.get("occupied_confirm_frames", 2)))
        self.empty_confirm_frames = max(1, int(temporal.get("empty_confirm_frames", 5)))
        self.sticky_occupied = bool(temporal.get("sticky_occupied", True))

        raw_slots = template.get("slots") if isinstance(template.get("slots"), list) else []
        self.slot_templates = [deepcopy(dict(slot)) for slot in raw_slots if isinstance(slot, Mapping)]
        self.slot_order = [str(item) for item in template.get("slot_order", [])]
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            self.tray_bbox = None  # type: Optional[BBox]
            self.tray_polygon = None  # type: Optional[Polygon]
            self.frame_count = 0
            self.slot_state = {
                str(slot["slot_id"]): {
                    "occupied": False,
                    "occupied_hits": 0,
                    "empty_hits": 0,
                    "matched_detection_id": None,
                }
                for slot in self.slot_templates
            }

    def _semantic(self, detection: Mapping[str, Any]) -> Optional[str]:
        class_id_value = detection.get("class_id")
        try:
            class_id = int(class_id_value) if class_id_value is not None and not isinstance(class_id_value, bool) else None
        except (TypeError, ValueError):
            class_id = None
        class_name = str(detection.get("class_name") or "").strip().lower()
        if (class_id is not None and class_id in self.tray_ids) or (class_name and class_name in self.tray_names):
            return "tray"
        if (class_id is not None and class_id in self.box_ids) or (class_name and class_name in self.box_names):
            return "box"
        return None

    def _accepted_detections(
        self, runtime_result: Mapping[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        trays = []  # type: List[Dict[str, Any]]
        boxes = []  # type: List[Dict[str, Any]]
        rejected_non_obb = 0
        detections = runtime_result.get("detections")
        for index, raw in enumerate(detections if isinstance(detections, list) else []):
            if not isinstance(raw, Mapping):
                continue
            semantic = self._semantic(raw)
            score = _number(raw.get("score"))
            box = _bbox(raw.get("bbox_xyxy"))
            polygon = _obb_polygon(raw)
            if semantic is None or box is None:
                continue
            if polygon is None:
                if self.require_obb:
                    rejected_non_obb += 1
                    continue
                polygon = _bbox_polygon(box)
            item = {
                "id": str(raw.get("id") or "det-{}".format(index)),
                "class_id": raw.get("class_id"),
                "class_name": str(raw.get("class_name") or ""),
                "score": score,
                "bbox": box,
                "polygon": polygon,
                "center": _center_from_detection(raw, polygon, box),
                "long_axis_angle_deg": _long_axis_angle(polygon),
            }
            if semantic == "tray" and score >= self.tray_min_confidence:
                trays.append(item)
            elif semantic == "box" and score >= self.box_min_confidence:
                boxes.append(item)
        return trays, boxes, rejected_non_obb

    def _update_tray(self, trays: Sequence[Mapping[str, Any]]) -> Optional[str]:
        if not trays:
            return "locked" if self.tray_polygon is not None and self.lock_after_first_detection else None
        selected = max(trays, key=lambda item: (float(item["score"]), _polygon_area(item["polygon"])))
        detected_polygon = list(selected["polygon"])
        detected_bbox = _polygon_bbox(detected_polygon)
        if self.tray_polygon is None or self.tray_bbox is None:
            self.tray_polygon = detected_polygon
            self.tray_bbox = detected_bbox
            return "detected"
        if bbox_iou(self.tray_bbox, detected_bbox) < self.update_min_iou:
            return "locked"
        alpha = self.ema_alpha
        self.tray_polygon = [
            (
                (1.0 - alpha) * old[0] + alpha * new[0],
                (1.0 - alpha) * old[1] + alpha * new[1],
            )
            for old, new in zip(self.tray_polygon, detected_polygon)
        ]
        self.tray_bbox = _polygon_bbox(self.tray_polygon)
        return "detected"

    def _build_slots(
        self, image_width: int, image_height: int
    ) -> Tuple[List[Dict[str, Any]], Polygon, Dict[str, float]]:
        assert self.tray_polygon is not None
        if self.footprint_mode == "centered_square_by_short_edge":
            footprint, footprint_meta = _centered_square_quad(
                self.tray_polygon, self.footprint_fill_ratio
            )
        else:
            footprint = list(self.tray_polygon)
            tray_width = (
                _edge_length(footprint[0], footprint[1])
                + _edge_length(footprint[3], footprint[2])
            ) / 2.0
            tray_height = (
                _edge_length(footprint[0], footprint[3])
                + _edge_length(footprint[1], footprint[2])
            ) / 2.0
            footprint_meta = {
                "u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0,
                "tray_width_px": tray_width, "tray_height_px": tray_height,
                "square_side_px": min(tray_width, tray_height),
            }

        footprint_angle = _edge_angle_deg(footprint[0], footprint[1])
        slots = []  # type: List[Dict[str, Any]]
        for template in self.slot_templates:
            polygon = []  # type: Polygon
            for nx, ny in template["polygon_norm"]:
                px, py = _bilinear_project(footprint, float(nx), float(ny))
                px = min(max(px, 0.0), float(max(0, image_width - 1)))
                py = min(max(py, 0.0), float(max(0, image_height - 1)))
                polygon.append((px, py))
            box = _polygon_bbox(polygon)
            template_orientation = float(template.get("orientation_deg", 0.0))
            slots.append(
                {
                    "slot_id": str(template["slot_id"]),
                    "name": str(template.get("name") or template["slot_id"]),
                    "template_orientation_deg": template_orientation,
                    "orientation_deg": _normalize_axis_angle(footprint_angle + template_orientation),
                    "orientation_label": "横向" if int(round(template_orientation / 90.0)) % 2 == 0 else "竖向",
                    "polygon": polygon,
                    "bbox": box,
                    "center": _polygon_center(polygon),
                }
            )
        return slots, footprint, footprint_meta

    def _match_boxes(
        self, slots: Sequence[Mapping[str, Any]], boxes: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        candidates = []  # type: List[Tuple[float, str, str, Dict[str, Any]]]
        for slot in slots:
            slot_id = str(slot["slot_id"])
            slot_box = slot["bbox"]  # type: BBox
            slot_center = slot["center"]  # type: Point
            slot_polygon = slot["polygon"]
            diagonal = max(1.0, math.hypot(slot_box[2] - slot_box[0], slot_box[3] - slot_box[1]))
            for item in boxes:
                box = item["bbox"]  # type: BBox
                center = item["center"]  # type: Point
                box_polygon = item["polygon"]
                inside = _point_in_polygon(center, slot_polygon)
                iou = polygon_iou(slot_polygon, box_polygon)
                distance_ratio = math.hypot(center[0] - slot_center[0], center[1] - slot_center[1]) / diagonal
                orientation_diff = _axis_angle_diff(
                    float(slot["orientation_deg"]), float(item["long_axis_angle_deg"])
                )
                if not (inside or iou >= self.min_iou):
                    continue
                if distance_ratio > self.max_center_distance_ratio:
                    continue
                if orientation_diff > self.max_orientation_diff_deg:
                    continue
                center_score = max(
                    0.0,
                    1.0 - distance_ratio / max(self.max_center_distance_ratio, 1e-6),
                )
                orientation_score = max(
                    0.0,
                    1.0 - orientation_diff / max(self.max_orientation_diff_deg, 1e-6),
                )
                score = (
                    (self.center_inside_bonus if inside else 0.0)
                    + self.iou_weight * iou
                    + self.center_weight * center_score
                    + self.orientation_weight * orientation_score
                    + 0.02 * float(item["score"])
                )
                detail = {
                    "detection_id": str(item["id"]),
                    "detection_bbox_xyxy": [round(value, 3) for value in box],
                    "detection_obb_points": [
                        [round(point_value[0], 3), round(point_value[1], 3)] for point_value in box_polygon
                    ],
                    "detection_center_xy": [round(value, 3) for value in center],
                    "polygon_iou": round(iou, 6),
                    "center_inside": inside,
                    "center_distance_ratio": round(distance_ratio, 6),
                    "orientation_diff_deg": round(orientation_diff, 3),
                    "match_score": round(score, 6),
                }
                candidates.append((score, slot_id, str(item["id"]), detail))

        matches = {}  # type: Dict[str, Dict[str, Any]]
        used_boxes = set()  # type: Set[str]
        for _, slot_id, detection_id, detail in sorted(candidates, reverse=True):
            if slot_id in matches or detection_id in used_boxes:
                continue
            matches[slot_id] = detail
            used_boxes.add(detection_id)
        return matches

    def _update_slot_states(self, matches: Mapping[str, Mapping[str, Any]]) -> None:
        for slot_id, state in self.slot_state.items():
            matched = matches.get(slot_id)
            if matched is not None:
                state["occupied_hits"] += 1
                state["empty_hits"] = 0
                state["matched_detection_id"] = matched["detection_id"]
                if state["occupied_hits"] >= self.occupied_confirm_frames:
                    state["occupied"] = True
            else:
                state["occupied_hits"] = 0
                state["empty_hits"] += 1
                if not self.sticky_occupied and state["empty_hits"] >= self.empty_confirm_frames:
                    state["occupied"] = False
                    state["matched_detection_id"] = None

    def evaluate(self, runtime_result: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.frame_count += 1
            image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
            image_width = max(1, int(_number(image.get("width"), 1)))
            image_height = max(1, int(_number(image.get("height"), 1)))
            trays, boxes, rejected_non_obb = self._accepted_detections(runtime_result)
            tray_source = self._update_tray(trays)

            if self.tray_bbox is None or self.tray_polygon is None or tray_source is None:
                return {
                    "layer": 1,
                    "state": "WAIT_TRAY",
                    "complete": False,
                    "slot_count": len(self.slot_templates),
                    "occupied_count": 0,
                    "empty_count": len(self.slot_templates),
                    "next_slot_id": None,
                    "tray": {"detected": False, "locked": False},
                    "slots": [],
                    "accepted_box_count": len(boxes),
                    "rejected_non_obb_count": rejected_non_obb,
                    "frame_count": self.frame_count,
                }

            slots, footprint, footprint_meta = self._build_slots(image_width, image_height)
            matches = self._match_boxes(slots, boxes)
            self._update_slot_states(matches)

            output_slots = []  # type: List[Dict[str, Any]]
            for slot in slots:
                slot_id = str(slot["slot_id"])
                state = self.slot_state[slot_id]
                match = matches.get(slot_id)
                output_slots.append(
                    {
                        "slot_id": slot_id,
                        "name": slot["name"],
                        "template_orientation_deg": slot["template_orientation_deg"],
                        "orientation_deg": round(float(slot["orientation_deg"]), 3),
                        "orientation_label": slot["orientation_label"],
                        "polygon": [[round(x, 3), round(y, 3)] for x, y in slot["polygon"]],
                        "bbox_xyxy": [round(value, 3) for value in slot["bbox"]],
                        "center_xy": [round(value, 3) for value in slot["center"]],
                        "occupied": bool(state["occupied"]),
                        "state": "OCCUPIED" if state["occupied"] else ("VERIFYING" if match else "EMPTY"),
                        "visible_mask": not bool(state["occupied"]),
                        "matched_detection_id": (
                            state["matched_detection_id"]
                            if state["occupied"]
                            else (match or {}).get("detection_id")
                        ),
                        "instant_match": dict(match) if match else None,
                        "occupied_hits": int(state["occupied_hits"]),
                        "empty_hits": int(state["empty_hits"]),
                    }
                )

            occupied = [slot["slot_id"] for slot in output_slots if slot["occupied"]]
            empty = [slot_id for slot_id in self.slot_order if slot_id not in occupied]
            complete = len(occupied) == len(output_slots) and bool(output_slots)
            next_slot_id = None if complete else (empty[0] if empty else None)
            state_name = "LAYER_1_COMPLETE" if complete else "LAYER_1_FILLING"
            tray_angle = _normalize_axis_angle(_edge_angle_deg(self.tray_polygon[0], self.tray_polygon[1]))
            return {
                "layer": 1,
                "state": state_name,
                "complete": complete,
                "slot_count": len(output_slots),
                "occupied_count": len(occupied),
                "empty_count": len(output_slots) - len(occupied),
                "occupied_slot_ids": occupied,
                "empty_slot_ids": empty,
                "next_slot_id": next_slot_id,
                "tray": {
                    "detected": bool(trays),
                    "locked": self.tray_polygon is not None,
                    "source": tray_source,
                    "bbox_xyxy": [round(value, 3) for value in self.tray_bbox],
                    "obb_points": [
                        [round(point_value[0], 3), round(point_value[1], 3)]
                        for point_value in self.tray_polygon
                    ],
                    "angle_deg": round(tray_angle, 3),
                },
                "footprint": {
                    "mode": self.footprint_mode,
                    "fill_ratio": round(self.footprint_fill_ratio, 4),
                    "obb_points": [
                        [round(point_value[0], 3), round(point_value[1], 3)]
                        for point_value in footprint
                    ],
                    "normalized_bounds": {
                        key: round(float(footprint_meta[key]), 6)
                        for key in ("u_min", "u_max", "v_min", "v_max")
                    },
                    "tray_width_px": round(float(footprint_meta["tray_width_px"]), 3),
                    "tray_height_px": round(float(footprint_meta["tray_height_px"]), 3),
                    "square_side_px": round(float(footprint_meta["square_side_px"]), 3),
                },
                "slots": output_slots,
                "accepted_box_count": len(boxes),
                "rejected_non_obb_count": rejected_non_obb,
                "instant_matched_slot_ids": sorted(matches),
                "frame_count": self.frame_count,
            }
