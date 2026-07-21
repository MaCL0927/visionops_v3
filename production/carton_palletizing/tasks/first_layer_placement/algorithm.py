"""OBB and RGB-D based multi-layer carton palletizing planning.

Layer 1 uses OBB carton detections. Layer 2 and above compare the current
D2C-aligned depth image with the completed previous layer. After each layer is
filled, several stable depth frames are captured as the next baseline. The same
state machine supports four layers or any configured layer count.

The module intentionally supports Python 3.8, which is the system Python on
LB3576 images used by this project.
"""

from __future__ import annotations

import math
import threading
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


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




def _scale_polygon(points: Sequence[Point], sx: float, sy: float) -> Polygon:
    return [(float(point[0]) * sx, float(point[1]) * sy) for point in points]


def _shrink_polygon(points: Sequence[Point], shrink_ratio: float) -> Polygon:
    ratio = min(0.45, max(0.0, float(shrink_ratio)))
    center = _polygon_center(points)
    scale = 1.0 - ratio
    return [
        (center[0] + (point[0] - center[0]) * scale, center[1] + (point[1] - center[1]) * scale)
        for point in points
    ]


def _polygon_mask(
    polygon: Sequence[Point],
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    shrink_ratio: float = 0.0,
) -> "np.ndarray":
    """Rasterize an RGB-space polygon into a depth-image boolean mask."""

    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    if len(polygon) < 3 or image_width <= 0 or image_height <= 0:
        return mask.astype(bool)
    shrunk = _shrink_polygon(polygon, shrink_ratio)
    sx = float(target_width) / float(image_width)
    sy = float(target_height) / float(image_height)
    scaled = _scale_polygon(shrunk, sx, sy)
    points = np.asarray(
        [
            [
                int(round(min(max(point[0], 0.0), float(max(0, target_width - 1))))),
                int(round(min(max(point[1], 0.0), float(max(0, target_height - 1))))),
            ]
            for point in scaled
        ],
        dtype=np.int32,
    )
    if points.shape[0] >= 3:
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _safe_percentile(values: "np.ndarray", percentile: float, default: float = 0.0) -> float:
    if values.size <= 0:
        return default
    result = float(np.percentile(values, percentile))
    return result if math.isfinite(result) else default


class MultiLayerPlacementAlgorithm:
    """Stateful OBB/RGB-D tracker for an arbitrary number of pallet layers."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = deepcopy(dict(settings))
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        geometry = settings.get("geometry") if isinstance(settings.get("geometry"), Mapping) else {}
        tracking = settings.get("tray_tracking") if isinstance(settings.get("tray_tracking"), Mapping) else {}
        matching = settings.get("matching") if isinstance(settings.get("matching"), Mapping) else {}
        temporal = settings.get("temporal") if isinstance(settings.get("temporal"), Mapping) else {}
        template = settings.get("template") if isinstance(settings.get("template"), Mapping) else {}
        layering = settings.get("layering") if isinstance(settings.get("layering"), Mapping) else {}
        depth = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}

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

        self.max_layers = max(0, int(layering.get("max_layers", 4)))
        self.auto_advance = bool(layering.get("auto_advance", True))
        self.baseline_capture_frames = max(1, int(layering.get("baseline_capture_frames", 3)))
        self.baseline_settle_frames = max(0, int(layering.get("baseline_settle_frames", 5)))
        self.baseline_stability_mm = max(0.0, float(layering.get("baseline_stability_mm", 15.0)))
        self.use_previous_detected_boxes = bool(layering.get("use_previous_detected_boxes", True))

        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))
        self.slot_roi_shrink_ratio = min(0.45, max(0.0, float(depth.get("slot_roi_shrink_ratio", 0.12))))
        self.min_valid_ratio = min(1.0, max(0.0, float(depth.get("min_valid_ratio", 0.45))))
        self.baseline_min_valid_ratio = min(
            1.0, max(0.0, float(depth.get("baseline_min_valid_ratio", 0.55)))
        )
        self.min_height_delta_mm = max(0.0, float(depth.get("min_height_delta_mm", 80.0)))
        self.max_height_delta_mm = max(
            self.min_height_delta_mm + 1.0, float(depth.get("max_height_delta_mm", 600.0))
        )
        self.min_coverage_ratio = min(1.0, max(0.0, float(depth.get("min_coverage_ratio", 0.55))))
        self.height_percentile = min(100.0, max(0.0, float(depth.get("height_percentile", 50.0))))
        self.depth_occupied_confirm_frames = max(
            1, int(depth.get("occupied_confirm_frames", self.occupied_confirm_frames))
        )
        self.depth_occupied_stability_mm = max(
            0.0, float(depth.get("occupied_stability_mm", 20.0))
        )

        self.layer_template_strategy = str(
            template.get("layer_strategy") or template.get("strategy") or "single"
        ).strip().lower()
        self.template_sets = {}  # type: Dict[str, List[Dict[str, Any]]]
        self.template_metadata = {}  # type: Dict[str, Dict[str, Any]]

        raw_template_sets = template.get("templates")
        if isinstance(raw_template_sets, Mapping):
            for raw_key, raw_value in raw_template_sets.items():
                key = str(raw_key).strip().lower()
                if not key:
                    continue
                if isinstance(raw_value, Mapping):
                    raw_slots = raw_value.get("slots")
                    metadata = {
                        str(meta_key): deepcopy(meta_value)
                        for meta_key, meta_value in raw_value.items()
                        if str(meta_key) != "slots"
                    }
                elif isinstance(raw_value, list):
                    raw_slots = raw_value
                    metadata = {}
                else:
                    continue
                slots = [
                    deepcopy(dict(slot))
                    for slot in (raw_slots if isinstance(raw_slots, list) else [])
                    if isinstance(slot, Mapping)
                ]
                if slots:
                    self.template_sets[key] = slots
                    self.template_metadata[key] = metadata

        # Backward compatibility with the original single ``template.slots``
        # schema.  A repository/config that has not yet adopted alternating
        # layers therefore keeps exactly the previous behaviour.
        legacy_slots = template.get("slots") if isinstance(template.get("slots"), list) else []
        if not self.template_sets:
            slots = [deepcopy(dict(slot)) for slot in legacy_slots if isinstance(slot, Mapping)]
            self.template_sets["default"] = slots
            self.template_metadata["default"] = {"template_id": "default"}
            self.layer_template_strategy = "single"

        self.default_template_key = str(template.get("default_template") or "").strip().lower()
        if self.default_template_key not in self.template_sets:
            self.default_template_key = next(iter(self.template_sets), "default")

        first_templates = self._slot_templates_for_layer(1)
        self.slot_templates = deepcopy(first_templates)
        self.slot_ids = [str(slot["slot_id"]) for slot in first_templates]
        expected_ids = set(self.slot_ids)
        for key, slots in self.template_sets.items():
            slot_ids = [str(slot["slot_id"]) for slot in slots]
            if len(slot_ids) != len(set(slot_ids)):
                raise ValueError("摆放模板 {!r} 存在重复 slot_id".format(key))
            if set(slot_ids) != expected_ids:
                raise ValueError(
                    "奇偶层摆放模板必须使用相同 slot_id，模板 {!r}={}，基准={}".format(
                        key, sorted(slot_ids), sorted(expected_ids)
                    )
                )

        self.slot_order = [str(item) for item in template.get("slot_order", [])]
        if not self.slot_order:
            self.slot_order = list(self.slot_ids)
        if set(self.slot_order) != expected_ids:
            raise ValueError(
                "template.slot_order 必须完整包含所有 slot_id，当前={}，需要={}".format(
                    self.slot_order, self.slot_ids
                )
            )

        configured_geometry_source = str(
            layering.get("next_layer_geometry")
            or layering.get("geometry_source")
            or ""
        ).strip().lower()
        if configured_geometry_source:
            self.next_layer_geometry = configured_geometry_source
        elif len(self.template_sets) > 1:
            self.next_layer_geometry = "layer_template"
        elif self.use_previous_detected_boxes:
            self.next_layer_geometry = "previous_layer_detected_boxes"
        else:
            self.next_layer_geometry = "previous_layer_slots"
        self._lock = threading.RLock()
        self.reset()

    def _template_key_for_layer(self, layer: Optional[int] = None) -> str:
        layer_number = self.current_layer if layer is None else max(1, int(layer))
        strategy = self.layer_template_strategy
        if strategy in {"odd_even", "alternating", "alternate"}:
            preferred = "odd" if layer_number % 2 == 1 else "even"
            if preferred in self.template_sets:
                return preferred
        layer_key = str(layer_number)
        if strategy in {"per_layer", "layer_number"} and layer_key in self.template_sets:
            return layer_key
        return self.default_template_key

    def _slot_templates_for_layer(self, layer: Optional[int] = None) -> List[Dict[str, Any]]:
        key = self._template_key_for_layer(layer)
        return self.template_sets.get(key) or self.template_sets[self.default_template_key]

    def _template_document(self, layer: Optional[int] = None) -> Dict[str, Any]:
        layer_number = self.current_layer if layer is None else max(1, int(layer))
        key = self._template_key_for_layer(layer_number)
        metadata = self.template_metadata.get(key, {})
        return {
            "strategy": self.layer_template_strategy,
            "key": key,
            "template_id": str(metadata.get("template_id") or key),
            "name": str(metadata.get("name") or key),
            "layer": layer_number,
            "geometry_source": self.next_layer_geometry,
        }

    def _new_slot_state(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(slot["slot_id"]): {
                "occupied": False,
                "occupied_hits": 0,
                "empty_hits": 0,
                "matched_detection_id": None,
                "last_detection_polygon": None,
                "last_depth": None,
            }
            for slot in self._slot_templates_for_layer(1)
        }

    def reset(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            self.tray_bbox = None  # type: Optional[BBox]
            self.tray_polygon = None  # type: Optional[Polygon]
            self.frame_count = 0
            self.current_layer = 1
            self.completed_layers = []  # type: List[int]
            self.slot_state = self._new_slot_state()
            self.current_slot_norm_polygons = None  # type: Optional[Dict[str, Polygon]]
            self.current_slot_source = "layer_template:{}".format(self._template_key_for_layer(1))
            self.depth_baseline = None  # type: Optional[np.ndarray]
            self.depth_baseline_layer = 0
            self.baseline_candidates = []  # type: List[np.ndarray]
            self.baseline_capture_valid_ratio = 0.0
            self.baseline_capture_stability_mm = None  # type: Optional[float]
            self.baseline_settle_count = 0
            self.last_image_size = None  # type: Optional[Tuple[int, int]]

    def needs_depth(self) -> bool:
        """Return whether the next evaluation should retrieve a depth frame."""
        with self._lock:
            return self.current_layer >= 2 or self._current_layer_complete()

    def _current_layer_complete(self) -> bool:
        return bool(self.slot_state) and all(bool(state["occupied"]) for state in self.slot_state.values())

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

    def detection_candidates(
        self,
        runtime_result: Mapping[str, Any],
        update_tray_reference: bool = False,
    ) -> Dict[str, Any]:
        """Return OBB tray/box candidates for trigger-only robot tasks.

        ``held_box_pose`` must inspect the current OBB detections without
        advancing the pallet occupancy state machine.  This helper reuses the
        exact class/score/OBB acceptance rules used by the placement algorithm
        and optionally refreshes only the locked tray reference.
        """

        with self._lock:
            trays, boxes, rejected_non_obb = self._accepted_detections(runtime_result)
            tray_source = None
            if update_tray_reference:
                tray_source = self._update_tray(trays)
            elif self.tray_polygon is not None:
                tray_source = "locked"
            elif trays:
                selected = max(trays, key=lambda item: (float(item["score"]), _polygon_area(item["polygon"])))
                tray_source = "detected_unlocked"
                tray_polygon = list(selected["polygon"])
                tray_bbox = _polygon_bbox(tray_polygon)
                return {
                    "trays": deepcopy(trays),
                    "boxes": deepcopy(boxes),
                    "rejected_non_obb_count": rejected_non_obb,
                    "tray_source": tray_source,
                    "tray_polygon": deepcopy(tray_polygon),
                    "tray_bbox": list(tray_bbox),
                }
            return {
                "trays": deepcopy(trays),
                "boxes": deepcopy(boxes),
                "rejected_non_obb_count": rejected_non_obb,
                "tray_source": tray_source,
                "tray_polygon": deepcopy(self.tray_polygon),
                "tray_bbox": list(self.tray_bbox) if self.tray_bbox is not None else None,
            }

    def tray_reference(self) -> Dict[str, Any]:
        """Return a thread-safe copy of the currently locked tray geometry."""

        with self._lock:
            return {
                "polygon": deepcopy(self.tray_polygon),
                "bbox": list(self.tray_bbox) if self.tray_bbox is not None else None,
                "layer": self.current_layer,
            }

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

    def _build_template_slots(
        self, image_width: int, image_height: int
    ) -> Tuple[List[Dict[str, Any]], Polygon, Dict[str, float]]:
        assert self.tray_polygon is not None
        if self.footprint_mode == "centered_square_by_short_edge":
            footprint, footprint_meta = _centered_square_quad(self.tray_polygon, self.footprint_fill_ratio)
        else:
            footprint = list(self.tray_polygon)
            tray_width = (
                _edge_length(footprint[0], footprint[1]) + _edge_length(footprint[3], footprint[2])
            ) / 2.0
            tray_height = (
                _edge_length(footprint[0], footprint[3]) + _edge_length(footprint[1], footprint[2])
            ) / 2.0
            footprint_meta = {
                "u_min": 0.0,
                "u_max": 1.0,
                "v_min": 0.0,
                "v_max": 1.0,
                "tray_width_px": tray_width,
                "tray_height_px": tray_height,
                "square_side_px": min(tray_width, tray_height),
            }

        footprint_angle = _edge_angle_deg(footprint[0], footprint[1])
        template_document = self._template_document()
        active_templates = self._slot_templates_for_layer()
        slots = []  # type: List[Dict[str, Any]]
        for template in active_templates:
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
                    "source": "layer_template:{}".format(template_document["key"]),
                    "template_key": template_document["key"],
                    "template_id": template_document["template_id"],
                }
            )
        return slots, footprint, footprint_meta

    def _build_slots(
        self, image_width: int, image_height: int
    ) -> Tuple[List[Dict[str, Any]], Polygon, Dict[str, float]]:
        template_slots, footprint, footprint_meta = self._build_template_slots(image_width, image_height)
        if (
            self.next_layer_geometry == "layer_template"
            or self.current_layer <= 1
            or not self.current_slot_norm_polygons
        ):
            return template_slots, footprint, footprint_meta

        template_by_id = {str(item["slot_id"]): item for item in template_slots}
        slots = []  # type: List[Dict[str, Any]]
        for slot_id in [str(slot["slot_id"]) for slot in self._slot_templates_for_layer()]:
            normalized = self.current_slot_norm_polygons.get(slot_id)
            fallback = template_by_id[slot_id]
            if not normalized or len(normalized) < 3:
                slots.append(fallback)
                continue
            polygon = [
                (
                    min(max(float(point[0]) * image_width, 0.0), float(max(0, image_width - 1))),
                    min(max(float(point[1]) * image_height, 0.0), float(max(0, image_height - 1))),
                )
                for point in normalized
            ]
            angle = _long_axis_angle(polygon)
            slots.append(
                {
                    "slot_id": slot_id,
                    "name": fallback["name"],
                    "template_orientation_deg": fallback["template_orientation_deg"],
                    "orientation_deg": angle,
                    "orientation_label": fallback["orientation_label"],
                    "polygon": polygon,
                    "bbox": _polygon_bbox(polygon),
                    "center": _polygon_center(polygon),
                    "source": self.current_slot_source,
                    "template_key": fallback.get("template_key"),
                    "template_id": fallback.get("template_id"),
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
                iou = polygon_iou(slot_polygon, box_polygon)
                inside = _point_in_polygon(center, slot_polygon)
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
                    0.0, 1.0 - distance_ratio / max(self.max_center_distance_ratio, 1e-6)
                )
                orientation_score = max(
                    0.0, 1.0 - orientation_diff / max(self.max_orientation_diff_deg, 1e-6)
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

    def _update_slot_states_rgb(self, matches: Mapping[str, Mapping[str, Any]]) -> None:
        for slot_id, state in self.slot_state.items():
            matched = matches.get(slot_id)
            if matched is not None:
                state["occupied_hits"] += 1
                state["empty_hits"] = 0
                state["matched_detection_id"] = matched["detection_id"]
                state["last_detection_polygon"] = deepcopy(matched.get("detection_obb_points"))
                if state["occupied_hits"] >= self.occupied_confirm_frames:
                    state["occupied"] = True
            else:
                state["occupied_hits"] = 0
                state["empty_hits"] += 1
                if not self.sticky_occupied and state["empty_hits"] >= self.empty_confirm_frames:
                    state["occupied"] = False
                    state["matched_detection_id"] = None

    def _depth_slot_result(
        self,
        slot: Mapping[str, Any],
        depth_image: "np.ndarray",
        image_width: int,
        image_height: int,
    ) -> Dict[str, Any]:
        if self.depth_baseline is None:
            return {"valid": False, "reason": "BASELINE_NOT_READY", "occupied": False}
        if depth_image.ndim != 2 or self.depth_baseline.shape != depth_image.shape:
            return {
                "valid": False,
                "reason": "DEPTH_SHAPE_MISMATCH",
                "occupied": False,
                "baseline_shape": list(self.depth_baseline.shape),
                "current_shape": list(depth_image.shape),
            }
        depth_height, depth_width = int(depth_image.shape[0]), int(depth_image.shape[1])
        mask = _polygon_mask(
            slot["polygon"],
            image_width,
            image_height,
            depth_width,
            depth_height,
            self.slot_roi_shrink_ratio,
        )
        roi_pixels = int(np.count_nonzero(mask))
        if roi_pixels <= 0:
            return {"valid": False, "reason": "EMPTY_ROI", "occupied": False, "roi_pixels": 0}
        baseline = self.depth_baseline
        valid = (
            mask
            & (baseline >= self.min_depth_mm)
            & (baseline <= self.max_depth_mm)
            & (depth_image >= self.min_depth_mm)
            & (depth_image <= self.max_depth_mm)
        )
        valid_count = int(np.count_nonzero(valid))
        valid_ratio = valid_count / float(roi_pixels)
        if valid_count <= 0:
            return {
                "valid": False,
                "reason": "NO_VALID_DEPTH",
                "occupied": False,
                "roi_pixels": roi_pixels,
                "valid_pixels": 0,
                "valid_ratio": 0.0,
            }
        delta = baseline[valid].astype(np.float32) - depth_image[valid].astype(np.float32)
        height_delta = _safe_percentile(delta, self.height_percentile)
        coverage = float(np.count_nonzero(delta >= self.min_height_delta_mm)) / float(valid_count)
        occupied = (
            valid_ratio >= self.min_valid_ratio
            and self.min_height_delta_mm <= height_delta <= self.max_height_delta_mm
            and coverage >= self.min_coverage_ratio
        )
        return {
            "valid": valid_ratio >= self.min_valid_ratio,
            "reason": "OK" if valid_ratio >= self.min_valid_ratio else "LOW_VALID_RATIO",
            "occupied": occupied,
            "roi_pixels": roi_pixels,
            "valid_pixels": valid_count,
            "valid_ratio": round(valid_ratio, 6),
            "height_delta_mm": round(height_delta, 3),
            "coverage_ratio": round(coverage, 6),
            "min_height_delta_mm": round(self.min_height_delta_mm, 3),
            "max_height_delta_mm": round(self.max_height_delta_mm, 3),
        }

    def _update_slot_states_depth(
        self,
        slots: Sequence[Mapping[str, Any]],
        depth_image: Optional["np.ndarray"],
        image_width: int,
        image_height: int,
        rgb_matches: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        details = {}  # type: Dict[str, Dict[str, Any]]
        if depth_image is None or self.depth_baseline is None:
            return details
        for slot in slots:
            slot_id = str(slot["slot_id"])
            state = self.slot_state[slot_id]
            detail = self._depth_slot_result(slot, depth_image, image_width, image_height)
            previous_depth = state.get("last_depth") if isinstance(state.get("last_depth"), Mapping) else None
            stable = False
            if bool(detail.get("occupied")) and previous_depth and bool(previous_depth.get("occupied")):
                current_delta = _number(detail.get("height_delta_mm"), float("nan"))
                previous_delta = _number(previous_depth.get("height_delta_mm"), float("nan"))
                if math.isfinite(current_delta) and math.isfinite(previous_delta):
                    stable = abs(current_delta - previous_delta) <= self.depth_occupied_stability_mm
            detail["stable_with_previous"] = stable
            detail["occupied_stability_mm"] = round(self.depth_occupied_stability_mm, 3)
            details[slot_id] = detail
            rgb_match = rgb_matches.get(slot_id)
            if rgb_match is not None:
                state["matched_detection_id"] = rgb_match.get("detection_id")
                state["last_detection_polygon"] = deepcopy(rgb_match.get("detection_obb_points"))
            if bool(detail.get("occupied")):
                state["occupied_hits"] = state["occupied_hits"] + 1 if stable else 1
                state["empty_hits"] = 0
                if state["occupied_hits"] >= self.depth_occupied_confirm_frames:
                    state["occupied"] = True
            else:
                state["occupied_hits"] = 0
                state["empty_hits"] += 1
                if not self.sticky_occupied and state["empty_hits"] >= self.empty_confirm_frames:
                    state["occupied"] = False
                    state["matched_detection_id"] = None
            state["last_depth"] = deepcopy(detail)
        return details

    def _footprint_mask(
        self,
        slots: Sequence[Mapping[str, Any]],
        image_width: int,
        image_height: int,
        depth_width: int,
        depth_height: int,
    ) -> "np.ndarray":
        union = np.zeros((depth_height, depth_width), dtype=bool)
        for slot in slots:
            union |= _polygon_mask(
                slot["polygon"], image_width, image_height, depth_width, depth_height, self.slot_roi_shrink_ratio
            )
        return union

    def _capture_depth_baseline(
        self,
        depth_image: "np.ndarray",
        slots: Sequence[Mapping[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        if depth_image.ndim != 2:
            self.baseline_candidates = []
            return False, {"valid": False, "reason": "DEPTH_DIMENSION_INVALID"}
        depth_height, depth_width = int(depth_image.shape[0]), int(depth_image.shape[1])
        mask = self._footprint_mask(slots, image_width, image_height, depth_width, depth_height)
        roi_pixels = int(np.count_nonzero(mask))
        valid = mask & (depth_image >= self.min_depth_mm) & (depth_image <= self.max_depth_mm)
        valid_count = int(np.count_nonzero(valid))
        valid_ratio = valid_count / float(max(1, roi_pixels))
        self.baseline_capture_valid_ratio = valid_ratio
        if valid_ratio < self.baseline_min_valid_ratio:
            self.baseline_candidates = []
            self.baseline_capture_stability_mm = None
            return False, {
                "valid": False,
                "reason": "LOW_VALID_RATIO",
                "valid_ratio": round(valid_ratio, 6),
                "required_valid_ratio": round(self.baseline_min_valid_ratio, 6),
                "captured_frames": 0,
                "required_frames": self.baseline_capture_frames,
            }

        candidate = depth_image.astype(np.uint16, copy=True)
        stability = 0.0
        if self.baseline_candidates:
            previous = self.baseline_candidates[-1]
            common = (
                mask
                & (previous >= self.min_depth_mm)
                & (previous <= self.max_depth_mm)
                & (candidate >= self.min_depth_mm)
                & (candidate <= self.max_depth_mm)
            )
            if np.any(common):
                difference = np.abs(previous[common].astype(np.float32) - candidate[common].astype(np.float32))
                stability = _safe_percentile(difference, 50.0)
            if stability > self.baseline_stability_mm:
                self.baseline_candidates = []
        self.baseline_capture_stability_mm = stability
        self.baseline_candidates.append(candidate)
        if len(self.baseline_candidates) < self.baseline_capture_frames:
            return False, {
                "valid": True,
                "reason": "CAPTURING",
                "valid_ratio": round(valid_ratio, 6),
                "stability_mm": round(stability, 3),
                "captured_frames": len(self.baseline_candidates),
                "required_frames": self.baseline_capture_frames,
            }

        stack = np.stack(self.baseline_candidates[-self.baseline_capture_frames :], axis=0).astype(np.float32)
        invalid = (stack < self.min_depth_mm) | (stack > self.max_depth_mm)
        stack[invalid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(all="ignore"):
                baseline = np.nanmedian(stack, axis=0)
        baseline = np.where(np.isfinite(baseline), baseline, 0.0).astype(np.uint16)
        self.depth_baseline = baseline
        self.depth_baseline_layer = self.current_layer
        self.baseline_candidates = []
        return True, {
            "valid": True,
            "reason": "READY",
            "valid_ratio": round(valid_ratio, 6),
            "stability_mm": round(stability, 3),
            "captured_frames": self.baseline_capture_frames,
            "required_frames": self.baseline_capture_frames,
        }

    def _derive_next_layer_geometry(
        self,
        slots: Sequence[Mapping[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Polygon]:
        output = {}  # type: Dict[str, Polygon]
        for slot in slots:
            slot_id = str(slot["slot_id"])
            state = self.slot_state.get(slot_id, {})
            points_value = state.get("last_detection_polygon") if self.use_previous_detected_boxes else None
            polygon = []  # type: Polygon
            if isinstance(points_value, list) and len(points_value) >= 3:
                for value in points_value:
                    parsed = _point(value)
                    if parsed is not None:
                        polygon.append(parsed)
            if len(polygon) < 3:
                polygon = list(slot["polygon"])
            output[slot_id] = [
                (
                    min(1.0, max(0.0, float(point[0]) / float(max(1, image_width)))),
                    min(1.0, max(0.0, float(point[1]) / float(max(1, image_height)))),
                )
                for point in polygon
            ]
        return output

    def _advance_layer(
        self,
        slots: Sequence[Mapping[str, Any]],
        image_width: int,
        image_height: int,
    ) -> None:
        finished = self.current_layer
        if finished not in self.completed_layers:
            self.completed_layers.append(finished)
        next_layer = finished + 1
        if self.next_layer_geometry == "layer_template":
            self.current_slot_norm_polygons = None
            self.current_slot_source = "layer_template:{}".format(
                self._template_key_for_layer(next_layer)
            )
        else:
            self.current_slot_norm_polygons = self._derive_next_layer_geometry(
                slots, image_width, image_height
            )
            self.current_slot_source = (
                "previous_layer_detected_boxes"
                if self.next_layer_geometry == "previous_layer_detected_boxes"
                else "previous_layer_slots"
            )
        self.current_layer = next_layer
        self.slot_state = self._new_slot_state()
        self.baseline_candidates = []
        self.baseline_settle_count = 0

    def _output_slots(
        self,
        slots: Sequence[Mapping[str, Any]],
        rgb_matches: Mapping[str, Mapping[str, Any]],
        depth_details: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        output = []  # type: List[Dict[str, Any]]
        for slot in slots:
            slot_id = str(slot["slot_id"])
            state = self.slot_state[slot_id]
            rgb_match = rgb_matches.get(slot_id)
            depth_detail = depth_details.get(slot_id)
            verifying = bool(rgb_match) if self.current_layer == 1 else bool(depth_detail and depth_detail.get("occupied"))
            output.append(
                {
                    "slot_id": slot_id,
                    "slot_key": "L{}:{}".format(self.current_layer, slot_id),
                    "name": slot["name"],
                    "source": slot.get("source"),
                    "template_key": slot.get("template_key"),
                    "template_id": slot.get("template_id"),
                    "template_orientation_deg": slot["template_orientation_deg"],
                    "orientation_deg": round(float(slot["orientation_deg"]), 3),
                    "orientation_label": slot["orientation_label"],
                    "polygon": [[round(x, 3), round(y, 3)] for x, y in slot["polygon"]],
                    "bbox_xyxy": [round(value, 3) for value in slot["bbox"]],
                    "center_xy": [round(value, 3) for value in slot["center"]],
                    "occupied": bool(state["occupied"]),
                    "state": "OCCUPIED" if state["occupied"] else ("VERIFYING" if verifying else "EMPTY"),
                    "visible_mask": not bool(state["occupied"]),
                    "matched_detection_id": state.get("matched_detection_id") or (rgb_match or {}).get("detection_id"),
                    "instant_match": dict(rgb_match) if rgb_match else None,
                    "depth": dict(depth_detail) if depth_detail else None,
                    "occupied_hits": int(state["occupied_hits"]),
                    "empty_hits": int(state["empty_hits"]),
                }
            )
        return output

    def _base_result(
        self,
        trays: Sequence[Mapping[str, Any]],
        boxes: Sequence[Mapping[str, Any]],
        rejected_non_obb: int,
        tray_source: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "layer": self.current_layer,
            "max_layers": self.max_layers,
            "completed_layers": list(self.completed_layers),
            "state": "WAIT_TRAY",
            "complete": False,
            "layer_complete": False,
            "stack_complete": False,
            "slot_count": len(self.slot_ids),
            "occupied_count": 0,
            "empty_count": len(self.slot_ids),
            "next_slot_id": None,
            "next_slot_key": None,
            "next_layer": None,
            "template": self._template_document(),
            "tray": {"detected": False, "locked": False, "source": tray_source},
            "slots": [],
            "accepted_box_count": len(boxes),
            "rejected_non_obb_count": rejected_non_obb,
            "frame_count": self.frame_count,
            "depth": {
                "required": self.current_layer >= 2,
                "available": False,
                "baseline_ready": self.depth_baseline is not None,
                "baseline_layer": self.depth_baseline_layer,
            },
        }

    def evaluate(
        self,
        runtime_result: Mapping[str, Any],
        depth_image: Optional["np.ndarray"] = None,
        depth_status: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self.frame_count += 1
            image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
            image_width = max(1, int(_number(image.get("width"), 1)))
            image_height = max(1, int(_number(image.get("height"), 1)))
            self.last_image_size = (image_width, image_height)
            if depth_image is not None:
                if not isinstance(depth_image, np.ndarray) or depth_image.ndim != 2:
                    raise ValueError("depth_image 必须是二维 numpy 数组")
                if depth_image.dtype != np.uint16:
                    depth_image = depth_image.astype(np.uint16, copy=False)

            trays, boxes, rejected_non_obb = self._accepted_detections(runtime_result)
            tray_source = self._update_tray(trays)
            if self.tray_bbox is None or self.tray_polygon is None or tray_source is None:
                return self._base_result(trays, boxes, rejected_non_obb, tray_source)

            slots, footprint, footprint_meta = self._build_slots(image_width, image_height)
            rgb_matches = self._match_boxes(slots, boxes)
            depth_details = {}  # type: Dict[str, Dict[str, Any]]
            if self.current_layer == 1:
                self._update_slot_states_rgb(rgb_matches)
            elif depth_image is not None and self.depth_baseline is not None:
                depth_details = self._update_slot_states_depth(
                    slots, depth_image, image_width, image_height, rgb_matches
                )

            output_slots = self._output_slots(slots, rgb_matches, depth_details)
            occupied = [slot["slot_id"] for slot in output_slots if slot["occupied"]]
            empty = [slot_id for slot_id in self.slot_order if slot_id not in occupied]
            layer_complete = len(occupied) == len(output_slots) and bool(output_slots)
            final_layer = self.max_layers > 0 and self.current_layer >= self.max_layers
            stack_complete = layer_complete and final_layer
            next_slot_id = None if layer_complete else (empty[0] if empty else None)
            next_layer = None
            transition = {
                "state": "NONE",
                "baseline_capture": {
                    "settled_frames": self.baseline_settle_count,
                    "required_settle_frames": self.baseline_settle_frames,
                    "captured_frames": len(self.baseline_candidates),
                    "required_frames": self.baseline_capture_frames,
                },
            }
            if not layer_complete:
                self.baseline_settle_count = 0
                self.baseline_candidates = []

            state_name = "LAYER_{}_FILLING".format(self.current_layer)
            depth_usable = (
                depth_image is not None
                and self.depth_baseline is not None
                and bool(depth_details)
                and any(bool(detail.get("valid")) for detail in depth_details.values())
            )
            if self.current_layer >= 2 and not depth_usable:
                state_name = "LAYER_{}_WAIT_DEPTH".format(self.current_layer)
            if layer_complete:
                if stack_complete:
                    if self.current_layer not in self.completed_layers:
                        self.completed_layers.append(self.current_layer)
                    state_name = "STACK_COMPLETE"
                    transition["state"] = "FINAL_LAYER_COMPLETE"
                elif not self.auto_advance:
                    state_name = "LAYER_{}_COMPLETE".format(self.current_layer)
                    transition["state"] = "MANUAL_ADVANCE_REQUIRED"
                elif depth_image is None:
                    state_name = "LAYER_{}_COMPLETE".format(self.current_layer)
                    transition["state"] = "WAIT_DEPTH_BASELINE"
                elif self.baseline_settle_count < self.baseline_settle_frames:
                    self.baseline_settle_count += 1
                    self.baseline_candidates = []
                    state_name = "LAYER_{}_SETTLING".format(self.current_layer)
                    transition["state"] = "WAIT_SETTLE"
                    transition["baseline_capture"] = {
                        "settled_frames": self.baseline_settle_count,
                        "required_settle_frames": self.baseline_settle_frames,
                        "captured_frames": 0,
                        "required_frames": self.baseline_capture_frames,
                    }
                else:
                    ready, capture = self._capture_depth_baseline(
                        depth_image, slots, image_width, image_height
                    )
                    capture["settled_frames"] = self.baseline_settle_count
                    capture["required_settle_frames"] = self.baseline_settle_frames
                    transition["baseline_capture"] = capture
                    if ready:
                        completed_layer = self.current_layer
                        self._advance_layer(slots, image_width, image_height)
                        next_layer = self.current_layer
                        new_slots, footprint, footprint_meta = self._build_slots(image_width, image_height)
                        output_slots = self._output_slots(new_slots, {}, {})
                        occupied = []
                        empty = list(self.slot_order)
                        layer_complete = False
                        stack_complete = False
                        next_slot_id = empty[0] if empty else None
                        state_name = "LAYER_{}_FILLING".format(self.current_layer)
                        transition["state"] = "ADVANCED"
                        transition["completed_layer"] = completed_layer
                        transition["next_layer"] = self.current_layer
                        slots = new_slots
                    else:
                        state_name = "LAYER_{}_CAPTURING_BASELINE".format(self.current_layer)
                        transition["state"] = "CAPTURING_BASELINE"

            tray_angle = _normalize_axis_angle(_edge_angle_deg(self.tray_polygon[0], self.tray_polygon[1]))
            depth_document = {
                "required": self.current_layer >= 2 or layer_complete,
                "available": depth_image is not None,
                "usable": depth_usable if self.current_layer >= 2 else depth_image is not None,
                "baseline_ready": self.depth_baseline is not None,
                "baseline_layer": self.depth_baseline_layer,
                "current_shape": list(depth_image.shape) if depth_image is not None else None,
                "min_height_delta_mm": round(self.min_height_delta_mm, 3),
                "max_height_delta_mm": round(self.max_height_delta_mm, 3),
                "min_coverage_ratio": round(self.min_coverage_ratio, 6),
            }
            if isinstance(depth_status, Mapping):
                depth_document["source"] = deepcopy(dict(depth_status))

            return {
                "layer": self.current_layer,
                "max_layers": self.max_layers,
                "completed_layers": list(self.completed_layers),
                "state": state_name,
                "complete": layer_complete,
                "layer_complete": layer_complete,
                "stack_complete": stack_complete,
                "slot_count": len(output_slots),
                "occupied_count": len(occupied),
                "empty_count": len(output_slots) - len(occupied),
                "occupied_slot_ids": occupied,
                "empty_slot_ids": empty,
                "next_slot_id": next_slot_id,
                "next_slot_key": (
                    "L{}:{}".format(self.current_layer, next_slot_id) if next_slot_id else None
                ),
                "next_layer": next_layer,
                "template": self._template_document(),
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
                "instant_matched_slot_ids": sorted(rgb_matches),
                "frame_count": self.frame_count,
                "depth": depth_document,
                "transition": transition,
            }


# Backward-compatible name used by the existing launcher and earlier tests.
FirstLayerPlacementAlgorithm = MultiLayerPlacementAlgorithm
