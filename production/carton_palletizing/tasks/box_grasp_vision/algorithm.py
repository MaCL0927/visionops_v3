#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert segmentation polygons into perspective carton grasp geometry.

The model supplies an instance mask polygon.  This module extracts a stable
quadrilateral, orders it as TL/TR/BR/BL, calculates the carton centre and the
midpoints of the left/right sides, and optionally samples aligned depth for all
robot-facing points.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

Point = Tuple[float, float]
Polygon = List[Point]


class SegmentationFormatError(ValueError):
    """Runtime segmentation payload does not contain a usable carton mask."""


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _class_id(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _point(value: object) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0], float("nan"))
    y = _number(value[1], float("nan"))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def _polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, current in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += current[0] * nxt[1] - nxt[0] * current[1]
    return abs(total) * 0.5


def _polygon_center(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    return (
        sum(item[0] for item in points) / float(len(points)),
        sum(item[1] for item in points) / float(len(points)),
    )


def _order_quad(points: Sequence[Point]) -> Polygon:
    """Order four image-space corners TL, TR, BR, BL.

    Sorting around the centroid is more stable than splitting only by y when
    the camera has a steep oblique view.  The cycle is then rotated to the
    top-left candidate and its direction is corrected so the next point lies on
    the right side.
    """
    if len(points) != 4:
        raise ValueError("quadrilateral must contain exactly four points")
    center = (
        sum(point[0] for point in points) / 4.0,
        sum(point[1] for point in points) / 4.0,
    )
    ordered = sorted(points, key=lambda item: math.atan2(item[1] - center[1], item[0] - center[0]))
    start = min(range(4), key=lambda index: (ordered[index][0] + ordered[index][1], ordered[index][1], ordered[index][0]))
    ordered = ordered[start:] + ordered[:start]
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0], ordered[-1], ordered[-2], ordered[-3]]
    return [(float(x), float(y)) for x, y in ordered]


def _cyclic_order(points: Sequence[Point]) -> Polygon:
    center = (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )
    return sorted(points, key=lambda item: math.atan2(item[1] - center[1], item[0] - center[0]))


def _best_four_from_vertices(vertices: Sequence[Point]) -> Optional[Polygon]:
    """Pick four cyclic hull vertices with maximum enclosed area."""
    if len(vertices) < 4:
        return None
    cyclic = _cyclic_order(vertices)
    if len(cyclic) == 4:
        return _order_quad(cyclic)
    # Approximation is capped before this function; combinations remain small.
    best = None  # type: Optional[Polygon]
    best_area = -1.0
    for indexes in itertools.combinations(range(len(cyclic)), 4):
        candidate = [cyclic[index] for index in indexes]
        area = _polygon_area(candidate)
        if area > best_area:
            best_area = area
            best = candidate
    return _order_quad(best) if best else None


def approximate_quadrilateral(
    contour: Sequence[Point],
    epsilon_min: float = 0.006,
    epsilon_max: float = 0.12,
    epsilon_steps: int = 28,
) -> Tuple[Polygon, Dict[str, Any]]:
    """Approximate a perspective mask contour by four corners.

    Exact four-point Douglas-Peucker approximations are preferred.  If none is
    found, the best small convex approximation is reduced to four area-maximising
    cyclic vertices.  ``minAreaRect`` is only the final safety fallback because
    it cannot preserve perspective trapezoids.
    """
    array = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if array.shape[0] < 4:
        raise SegmentationFormatError("mask contour contains fewer than four points")
    hull = cv2.convexHull(array)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1.0:
        raise SegmentationFormatError("mask contour perimeter is too small")

    exact = None  # type: Optional[np.ndarray]
    candidates = []  # type: List[Tuple[int, float, np.ndarray]]
    factors = np.linspace(float(epsilon_min), float(epsilon_max), max(2, int(epsilon_steps)))
    for factor in factors:
        approx = cv2.approxPolyDP(hull, float(factor) * perimeter, True)
        count = int(len(approx))
        candidates.append((count, float(factor), approx))
        if count == 4:
            exact = approx
            break

    method = "approx_poly_dp"
    epsilon_factor = 0.0
    if exact is not None:
        chosen = exact.reshape((-1, 2))
        epsilon_factor = next(item[1] for item in candidates if item[2] is exact)
        quad = _order_quad([(float(item[0]), float(item[1])) for item in chosen])
    else:
        # Prefer 5..12 point candidates; they retain perspective while limiting
        # the combinatorial reduction cost.
        useful = [item for item in candidates if 4 < item[0] <= 12]
        if useful:
            count, epsilon_factor, approx = min(useful, key=lambda item: (item[0] - 4, -item[1]))
            points = [(float(item[0][0]), float(item[0][1])) for item in approx]
            reduced = _best_four_from_vertices(points)
            if reduced is None:
                raise SegmentationFormatError("failed to reduce contour to four vertices")
            quad = reduced
            method = "convex_vertex_reduction"
        else:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            quad = _order_quad([(float(item[0]), float(item[1])) for item in box])
            method = "min_area_rect_fallback"

    contour_area = float(cv2.contourArea(hull))
    quad_area = _polygon_area(quad)
    area_ratio = quad_area / contour_area if contour_area > 1e-6 else 0.0
    return quad, {
        "method": method,
        "epsilon_factor": round(float(epsilon_factor), 6),
        "contour_points": int(array.shape[0]),
        "hull_points": int(hull.shape[0]),
        "contour_area_px": round(contour_area, 3),
        "quad_area_px": round(quad_area, 3),
        "quad_to_contour_area_ratio": round(area_ratio, 6),
    }


def _mask_rings(detection: Mapping[str, Any]) -> List[Polygon]:
    mask = detection.get("mask") if isinstance(detection.get("mask"), Mapping) else {}
    polygon = mask.get("polygon")
    if not isinstance(polygon, list):
        return []
    rings_raw = polygon
    if polygon and isinstance(polygon[0], (list, tuple)) and len(polygon[0]) >= 2 and isinstance(polygon[0][0], (int, float)):
        rings_raw = [polygon]
    rings = []  # type: List[Polygon]
    for raw_ring in rings_raw:
        if not isinstance(raw_ring, list):
            continue
        ring = []  # type: Polygon
        for raw_point in raw_ring:
            parsed = _point(raw_point)
            if parsed is not None:
                ring.append(parsed)
        if len(ring) >= 4 and _polygon_area(ring) > 1.0:
            rings.append(ring)
    return rings


def _round_point(point: Point, digits: int = 3) -> List[float]:
    return [round(float(point[0]), digits), round(float(point[1]), digits)]


def _midpoint(left: Point, right: Point) -> Point:
    return (left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5


def _move_toward(point: Point, center: Point, ratio: float) -> Point:
    safe = min(0.45, max(0.0, float(ratio)))
    return point[0] + (center[0] - point[0]) * safe, point[1] + (center[1] - point[1]) * safe


@dataclass
class ClassifiedMasks:
    image_width: int
    image_height: int
    items: List[Dict[str, Any]]
    ignored: List[Dict[str, Any]]


class BoxGraspAlgorithm:
    """Segmentation-to-grasp geometry processor for one or more cartons."""

    POINT_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left", "center", "left_mid", "right_mid")

    def __init__(self, settings: Mapping[str, Any]) -> None:
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        geometry = settings.get("geometry") if isinstance(settings.get("geometry"), Mapping) else {}
        depth = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}
        selection = settings.get("selection") if isinstance(settings.get("selection"), Mapping) else {}
        image = settings.get("image") if isinstance(settings.get("image"), Mapping) else {}

        self.box_ids = {int(item) for item in classes.get("box_class_ids", [0])}
        self.box_names = {str(item).strip().lower() for item in classes.get("box_class_names", ["box", "carton"]) if str(item).strip()}
        self.min_confidence = float(classes.get("box_min_confidence", 0.5))
        self.require_proto_mask = bool(geometry.get("require_proto_mask", True))
        self.min_mask_area_px = float(geometry.get("min_mask_area_px", 1500.0))
        self.epsilon_min = float(geometry.get("epsilon_min", 0.006))
        self.epsilon_max = float(geometry.get("epsilon_max", 0.12))
        self.epsilon_steps = int(geometry.get("epsilon_steps", 28))
        self.min_quad_area_ratio = float(geometry.get("min_quad_area_ratio", 0.65))
        self.max_quad_area_ratio = float(geometry.get("max_quad_area_ratio", 1.35))
        self.contour_max_points = max(4, int(geometry.get("contour_max_points", 160)))
        # Robot-facing grasp points are not kept exactly on the segmentation
        # boundary.  Move the left/right side midpoints toward the carton centre
        # so small mask jitter or perspective fitting errors cannot place a grasp
        # point outside the top surface.  The ratio is measured from the side
        # midpoint to the centre: 0 keeps the edge midpoint, 1 reaches the centre.
        self.grasp_inward_ratio = min(0.45, max(0.0, float(geometry.get("grasp_inward_ratio", 0.18))))
        self.max_targets = max(1, int(selection.get("max_targets", 1)))
        self.output_order = str(selection.get("output_order", "confidence")).strip().lower()

        self.expected_width = max(1, int(image.get("width", 640)))
        self.expected_height = max(1, int(image.get("height", 480)))
        self.require_fixed_size = bool(image.get("require_fixed_size", True))

        self.depth_enabled = bool(depth.get("enabled", True))
        self.depth_radius_px = max(0, int(depth.get("roi_radius_px", 4)))
        self.depth_percentile = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
        self.depth_min_valid_pixels = max(1, int(depth.get("min_valid_pixels", 3)))
        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))
        self.depth_inward_ratio = min(0.45, max(0.0, float(depth.get("edge_inward_ratio", 0.08))))
        # Sample depth slightly farther inside than the robot-facing grasp point,
        # while still projecting the sampled depth at the actual grasp coordinate.
        # On a planar carton top this avoids zero depth at a noisy mask boundary
        # without changing the point sent to the robot.
        self.grasp_depth_extra_inward_ratio = min(
            0.45,
            max(0.0, float(depth.get("grasp_extra_inward_ratio", 0.05))),
        )

    @staticmethod
    def _image_size(runtime_result: Mapping[str, Any]) -> Tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        width = int(_number(image.get("width")))
        height = int(_number(image.get("height")))
        if width <= 0 or height <= 0:
            raise SegmentationFormatError("Runtime inference_result lacks valid image.width/image.height")
        return width, height

    def _is_box(self, detection: Mapping[str, Any]) -> bool:
        cid = _class_id(detection.get("class_id"))
        name = str(detection.get("class_name") or "").strip().lower()
        return cid in self.box_ids or (bool(self.box_names) and name in self.box_names)

    @staticmethod
    def _simplify_contour(points: Polygon, max_points: int) -> Polygon:
        if len(points) <= max_points:
            return points
        step = float(len(points)) / float(max_points)
        return [points[min(len(points) - 1, int(round(index * step)))] for index in range(max_points)]

    def classify(self, runtime_result: Mapping[str, Any]) -> ClassifiedMasks:
        width, height = self._image_size(runtime_result)
        if self.require_fixed_size and (width != self.expected_width or height != self.expected_height):
            raise SegmentationFormatError(
                "box_grasp_vision expects {}x{}, Runtime returned {}x{}".format(
                    self.expected_width, self.expected_height, width, height
                )
            )
        detections = runtime_result.get("detections") if isinstance(runtime_result.get("detections"), list) else []
        accepted = []  # type: List[Dict[str, Any]]
        ignored = []  # type: List[Dict[str, Any]]
        for index, raw in enumerate(detections):
            if not isinstance(raw, Mapping):
                continue
            source_id = str(raw.get("id") or "seg-{}".format(index))
            if not self._is_box(raw):
                ignored.append({"id": source_id, "reason": "class_not_used"})
                continue
            score = _number(raw.get("score"))
            if score < self.min_confidence:
                ignored.append({"id": source_id, "reason": "low_confidence", "score": score})
                continue
            mask = raw.get("mask") if isinstance(raw.get("mask"), Mapping) else {}
            source = str(mask.get("source") or "")
            if self.require_proto_mask and source == "bbox_fallback":
                ignored.append({"id": source_id, "reason": "bbox_fallback_mask"})
                continue
            rings = _mask_rings(raw)
            if not rings:
                ignored.append({"id": source_id, "reason": "missing_polygon_mask"})
                continue
            contour = max(rings, key=_polygon_area)
            contour_area = _polygon_area(contour)
            if contour_area < self.min_mask_area_px:
                ignored.append({"id": source_id, "reason": "mask_too_small", "area_px": contour_area})
                continue
            try:
                quad, quality = approximate_quadrilateral(
                    contour,
                    self.epsilon_min,
                    self.epsilon_max,
                    self.epsilon_steps,
                )
            except Exception as error:
                ignored.append({"id": source_id, "reason": "quadrilateral_failed", "message": str(error)})
                continue
            area_ratio = float(quality.get("quad_to_contour_area_ratio", 0.0))
            if not self.min_quad_area_ratio <= area_ratio <= self.max_quad_area_ratio:
                ignored.append({"id": source_id, "reason": "quadrilateral_area_ratio", "ratio": area_ratio})
                continue
            top_left, top_right, bottom_right, bottom_left = quad
            left_edge_mid = _midpoint(top_left, bottom_left)
            right_edge_mid = _midpoint(top_right, bottom_right)
            center = _midpoint(left_edge_mid, right_edge_mid)

            # Move the actual robot-facing points inward along the line joining
            # the two side midpoints.  This guarantees the left point moves toward
            # the right and the right point moves toward the left, even for a
            # perspective trapezoid whose side midpoints have different y values.
            left_mid = _move_toward(left_edge_mid, center, self.grasp_inward_ratio)
            right_mid = _move_toward(right_edge_mid, center, self.grasp_inward_ratio)
            contour_center = _polygon_center(contour)
            points = {
                "top_left": top_left,
                "top_right": top_right,
                "bottom_right": bottom_right,
                "bottom_left": bottom_left,
                "center": center,
                "left_mid": left_mid,
                "right_mid": right_mid,
            }
            sample_points = {
                "top_left": _move_toward(top_left, center, self.depth_inward_ratio),
                "top_right": _move_toward(top_right, center, self.depth_inward_ratio),
                "bottom_right": _move_toward(bottom_right, center, self.depth_inward_ratio),
                "bottom_left": _move_toward(bottom_left, center, self.depth_inward_ratio),
                "center": center,
                "left_mid": _move_toward(left_mid, center, self.grasp_depth_extra_inward_ratio),
                "right_mid": _move_toward(right_mid, center, self.grasp_depth_extra_inward_ratio),
            }
            accepted.append({
                "source_id": source_id,
                "class_id": _class_id(raw.get("class_id")) if _class_id(raw.get("class_id")) is not None else 0,
                "class_name": str(raw.get("class_name") or "box"),
                "confidence": score,
                "bbox_xyxy": list(raw.get("bbox_xyxy") or []),
                "contour": self._simplify_contour(contour, self.contour_max_points),
                "quad": quad,
                "points": points,
                "depth_sample_points": sample_points,
                "edge_midpoints": {
                    "left_mid": left_edge_mid,
                    "right_mid": right_edge_mid,
                },
                "grasp_inward_ratio": self.grasp_inward_ratio,
                "grasp_depth_extra_inward_ratio": self.grasp_depth_extra_inward_ratio,
                "contour_center": contour_center,
                "quality": quality,
            })

        if self.output_order == "left_to_right":
            accepted.sort(key=lambda item: float(item["points"]["center"][0]))
        elif self.output_order == "top_to_bottom":
            accepted.sort(key=lambda item: float(item["points"]["center"][1]))
        else:
            accepted.sort(key=lambda item: -float(item["confidence"]))
        return ClassifiedMasks(width, height, accepted[: self.max_targets], ignored)

    @staticmethod
    def _map_pixel(value: float, source_size: int, target_size: int) -> int:
        if source_size <= 1 or target_size <= 1:
            return max(0, min(target_size - 1, int(round(value))))
        mapped = value * float(target_size - 1) / float(source_size - 1)
        return max(0, min(target_size - 1, int(round(mapped))))

    def sample_depth(self, depth: "np.ndarray", image_width: int, image_height: int, point: Point) -> Dict[str, Any]:
        height, width = int(depth.shape[0]), int(depth.shape[1])
        x = self._map_pixel(point[0], image_width, width)
        y = self._map_pixel(point[1], image_height, height)
        radius = self.depth_radius_px
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        roi = depth[y1:y2, x1:x2]
        valid = roi[(roi >= self.min_depth_mm) & (roi <= self.max_depth_mm)]
        if int(valid.size) < self.depth_min_valid_pixels:
            return {"depth_valid": False, "depth_mm": 0, "sample_px": [x, y], "valid_pixels": int(valid.size)}
        value = int(round(float(np.percentile(valid.astype(np.float32), self.depth_percentile))))
        return {"depth_valid": True, "depth_mm": value, "sample_px": [x, y], "valid_pixels": int(valid.size)}

    def sample_item_depth(self, item: Mapping[str, Any], depth: "np.ndarray", image_width: int, image_height: int) -> Dict[str, Dict[str, Any]]:
        output = {}  # type: Dict[str, Dict[str, Any]]
        sample_points = item.get("depth_sample_points") if isinstance(item.get("depth_sample_points"), Mapping) else {}
        for name in self.POINT_ORDER:
            raw = sample_points.get(name)
            parsed = _point(raw)
            if parsed is None:
                output[name] = {"depth_valid": False, "depth_mm": 0, "sample_px": [0, 0], "valid_pixels": 0}
            else:
                output[name] = self.sample_depth(depth, image_width, image_height, parsed)
        return output

    def build_deproject_input(self, item: Mapping[str, Any], depth_info: Mapping[str, Mapping[str, Any]]) -> List[List[float]]:
        points = item.get("points") if isinstance(item.get("points"), Mapping) else {}
        output = []  # type: List[List[float]]
        for name in self.POINT_ORDER:
            parsed = _point(points.get(name)) or (0.0, 0.0)
            info = depth_info.get(name) if isinstance(depth_info.get(name), Mapping) else {}
            depth_mm = int(info.get("depth_mm") or 0) if bool(info.get("depth_valid")) else 0
            output.append([float(parsed[0]), float(parsed[1]), float(depth_mm)])
        return output

    def build_external_item(
        self,
        item_id: int,
        item: Mapping[str, Any],
        depth_info: Mapping[str, Mapping[str, Any]],
        positions_camera: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        points = item.get("points") if isinstance(item.get("points"), Mapping) else {}
        camera_by_name = {}  # type: Dict[str, List[float]]
        depth_by_name = {}  # type: Dict[str, Dict[str, Any]]
        for index, name in enumerate(self.POINT_ORDER):
            position = positions_camera[index] if index < len(positions_camera) else [0.0, 0.0, 0.0]
            if not isinstance(position, (list, tuple)) or len(position) < 3:
                position = [0.0, 0.0, 0.0]
            camera_by_name[name] = [round(_number(position[0]), 3), round(_number(position[1]), 3), round(_number(position[2]), 3)]
            raw_depth = depth_info.get(name) if isinstance(depth_info.get(name), Mapping) else {}
            depth_by_name[name] = {
                "valid": bool(raw_depth.get("depth_valid")),
                "depth_mm": int(raw_depth.get("depth_mm") or 0),
                "sample_px": list(raw_depth.get("sample_px") or [0, 0]),
                "valid_pixels": int(raw_depth.get("valid_pixels") or 0),
            }

        corners_px = [_round_point(points.get(name) or (0.0, 0.0)) for name in self.POINT_ORDER[:4]]
        corners_camera = [camera_by_name[name] for name in self.POINT_ORDER[:4]]
        edge_midpoints = item.get("edge_midpoints") if isinstance(item.get("edge_midpoints"), Mapping) else {}
        return {
            "id": int(item_id),
            "source_id": str(item.get("source_id") or ""),
            "class_id": int(item.get("class_id") or 0),
            "class_name": str(item.get("class_name") or "box"),
            "confidence": round(float(item.get("confidence") or 0.0), 6),
            "contour_px": [_round_point(point) for point in item.get("contour", [])],
            "corners_px": {
                "top_left": corners_px[0],
                "top_right": corners_px[1],
                "bottom_right": corners_px[2],
                "bottom_left": corners_px[3],
            },
            "center_px": _round_point(points.get("center") or (0.0, 0.0)),
            "grasp_points_px": {
                "left_mid": _round_point(points.get("left_mid") or (0.0, 0.0)),
                "right_mid": _round_point(points.get("right_mid") or (0.0, 0.0)),
            },
            "grasp_geometry": {
                "edge_midpoints_px": {
                    "left_mid": _round_point(edge_midpoints.get("left_mid") or (0.0, 0.0)),
                    "right_mid": _round_point(edge_midpoints.get("right_mid") or (0.0, 0.0)),
                },
                "inward_ratio": round(float(item.get("grasp_inward_ratio") or 0.0), 6),
                "depth_extra_inward_ratio": round(float(item.get("grasp_depth_extra_inward_ratio") or 0.0), 6),
            },
            "corners_camera": {
                "top_left": corners_camera[0],
                "top_right": corners_camera[1],
                "bottom_right": corners_camera[2],
                "bottom_left": corners_camera[3],
            },
            "center_camera": camera_by_name["center"],
            "grasp_points_camera": {
                "left_mid": camera_by_name["left_mid"],
                "right_mid": camera_by_name["right_mid"],
            },
            "depth": depth_by_name,
            "quadrilateral_quality": dict(item.get("quality") or {}),
        }
