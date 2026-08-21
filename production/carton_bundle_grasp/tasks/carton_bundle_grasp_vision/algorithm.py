#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M41 carton-bundle top-plane reconstruction.

The segmentation model is used only as the semantic/top-surface anchor.  RGB-D
samples from the interior of the mask are robustly fitted to one 3-D plane in
camera coordinates.  The visible quadrilateral corners are then intersected
with that plane through SDK-derived camera rays.  Finally the observed plane
rectangle is regularised with the physical 715 x 525 mm bundle prior and the
midpoints of the two 525 mm width edges are returned.

No robot pitch/waist compensation is applied here.  All output XYZ values remain
in the camera frame and are suitable for the normal hand-eye transform on the
robot side.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

Point2 = Tuple[float, float]
Polygon = List[Point2]


class GeometryError(ValueError):
    """The mask/depth observation cannot support a reliable M41 geometry."""


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


def _point2(value: object) -> Optional[Point2]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0], float("nan"))
    y = _number(value[1], float("nan"))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def _vec3(value: object) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        arr = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _polygon_area(points: Sequence[Point2]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, current in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += current[0] * nxt[1] - nxt[0] * current[1]
    return abs(total) * 0.5


def _polygon_center(points: Sequence[Point2]) -> Point2:
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


def _order_quad(points: Sequence[Point2]) -> Polygon:
    if len(points) != 4:
        raise ValueError("quadrilateral must contain exactly four points")
    center = (
        sum(point[0] for point in points) / 4.0,
        sum(point[1] for point in points) / 4.0,
    )
    ordered = sorted(points, key=lambda item: math.atan2(item[1] - center[1], item[0] - center[0]))
    start = min(
        range(4),
        key=lambda index: (ordered[index][0] + ordered[index][1], ordered[index][1], ordered[index][0]),
    )
    ordered = ordered[start:] + ordered[:start]
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0], ordered[-1], ordered[-2], ordered[-3]]
    return [(float(x), float(y)) for x, y in ordered]


def _cyclic_order(points: Sequence[Point2]) -> Polygon:
    center = (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )
    return sorted(points, key=lambda item: math.atan2(item[1] - center[1], item[0] - center[0]))


def _best_four_from_vertices(vertices: Sequence[Point2]) -> Optional[Polygon]:
    if len(vertices) < 4:
        return None
    cyclic = _cyclic_order(vertices)
    if len(cyclic) == 4:
        return _order_quad(cyclic)
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
    contour: Sequence[Point2],
    epsilon_min: float = 0.006,
    epsilon_max: float = 0.12,
    epsilon_steps: int = 28,
) -> Tuple[Polygon, Dict[str, Any]]:
    array = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if array.shape[0] < 4:
        raise GeometryError("mask contour contains fewer than four points")
    hull = cv2.convexHull(array)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1.0:
        raise GeometryError("mask contour perimeter is too small")

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
        useful = [item for item in candidates if 4 < item[0] <= 12]
        if useful:
            _count, epsilon_factor, approx = min(useful, key=lambda item: (item[0] - 4, -item[1]))
            points = [(float(item[0][0]), float(item[0][1])) for item in approx]
            reduced = _best_four_from_vertices(points)
            if reduced is None:
                raise GeometryError("failed to reduce contour to four vertices")
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
            parsed = _point2(raw_point)
            if parsed is not None:
                ring.append(parsed)
        if len(ring) >= 4 and _polygon_area(ring) > 1.0:
            rings.append(ring)
    return rings


def _round2(point: Sequence[float], digits: int = 3) -> List[float]:
    return [round(float(point[0]), digits), round(float(point[1]), digits)]


def _round3(point: Sequence[float], digits: int = 3) -> List[float]:
    return [round(float(point[0]), digits), round(float(point[1]), digits), round(float(point[2]), digits)]


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise GeometryError("{} vector is degenerate".format(name))
    return vector / norm


@dataclass
class ClassifiedMasks:
    image_width: int
    image_height: int
    items: List[Dict[str, Any]]
    ignored: List[Dict[str, Any]]


class CartonBundleGraspAlgorithm:
    """M41 full-top bundle reconstruction and fixed-size regularisation."""

    CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
    GRASP_NAMES = ("width_mid_a", "width_mid_b")

    def __init__(self, settings: Mapping[str, Any]) -> None:
        image = settings.get("image") if isinstance(settings.get("image"), Mapping) else {}
        classes = settings.get("classes") if isinstance(settings.get("classes"), Mapping) else {}
        selection = settings.get("selection") if isinstance(settings.get("selection"), Mapping) else {}
        geometry = settings.get("geometry") if isinstance(settings.get("geometry"), Mapping) else {}
        depth = settings.get("depth") if isinstance(settings.get("depth"), Mapping) else {}
        prior = settings.get("bundle_prior") if isinstance(settings.get("bundle_prior"), Mapping) else {}
        plane = settings.get("top_plane") if isinstance(settings.get("top_plane"), Mapping) else {}

        self.expected_width = max(1, int(image.get("width", 1280)))
        self.expected_height = max(1, int(image.get("height", 720)))
        self.require_fixed_size = bool(image.get("require_fixed_size", False))

        self.target_ids = {int(item) for item in classes.get("target_ids", [0])}
        self.target_names = {
            str(item).strip().lower()
            for item in classes.get("target_names", ["carton_bundle_top", "bundle_top", "carton_bundle"])
            if str(item).strip()
        }
        self.min_confidence = float(classes.get("min_confidence", 0.5))
        self.max_targets = max(1, int(selection.get("max_targets", 1)))
        self.output_order = str(selection.get("mode", "confidence")).strip().lower()

        self.require_proto_mask = bool(geometry.get("require_proto_mask", True))
        self.min_mask_area_px = float(geometry.get("min_mask_area_px", 2500.0))
        self.epsilon_min = float(geometry.get("epsilon_min", 0.006))
        self.epsilon_max = float(geometry.get("epsilon_max", 0.12))
        self.epsilon_steps = max(2, int(geometry.get("epsilon_steps", 28)))
        self.min_quad_area_ratio = float(geometry.get("min_quad_area_ratio", 0.62))
        self.max_quad_area_ratio = float(geometry.get("max_quad_area_ratio", 1.38))
        self.contour_max_points = max(4, int(geometry.get("contour_max_points", 96)))

        self.length_mm = float(prior.get("length_mm", 715.0))
        self.width_mm = float(prior.get("width_mm", 525.0))
        if self.length_mm <= self.width_mm or self.width_mm <= 0.0:
            raise ValueError("bundle_prior requires length_mm > width_mm > 0")
        self.length_tolerance_mm = max(1.0, float(prior.get("length_tolerance_mm", 80.0)))
        self.width_tolerance_mm = max(1.0, float(prior.get("width_tolerance_mm", 70.0)))
        self.regularize_fixed_size = bool(prior.get("regularize_fixed_size", True))

        self.depth_enabled = bool(depth.get("enabled", True))
        self.depth_radius_px = max(0, int(depth.get("roi_radius_px", 2)))
        self.depth_percentile = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
        self.depth_min_valid_pixels = max(1, int(depth.get("min_valid_pixels", 1)))
        self.min_depth_mm = max(0, int(depth.get("min_depth_mm", 100)))
        self.max_depth_mm = max(self.min_depth_mm + 1, int(depth.get("max_depth_mm", 5000)))

        self.plane_sample_count = min(256, max(24, int(plane.get("sample_count", 96))))
        self.plane_erode_px = max(0, int(plane.get("erode_px", 18)))
        self.plane_ransac_threshold_mm = max(0.5, float(plane.get("ransac_threshold_mm", 5.0)))
        self.plane_ransac_trials = min(512, max(16, int(plane.get("ransac_trials", 96))))
        self.plane_min_valid_samples = max(12, int(plane.get("min_valid_samples", 36)))
        self.plane_min_inlier_ratio = min(1.0, max(0.1, float(plane.get("min_inlier_ratio", 0.70))))
        self.plane_max_rms_mm = max(0.5, float(plane.get("max_rms_mm", 6.0)))

    @staticmethod
    def _image_size(runtime_result: Mapping[str, Any]) -> Tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        width = int(_number(image.get("width")))
        height = int(_number(image.get("height")))
        if width <= 0 or height <= 0:
            raise GeometryError("Runtime inference_result lacks valid image.width/image.height")
        return width, height

    def _is_target(self, detection: Mapping[str, Any]) -> bool:
        cid = _class_id(detection.get("class_id"))
        name = str(detection.get("class_name") or "").strip().lower()
        return cid in self.target_ids or (bool(self.target_names) and name in self.target_names)

    @staticmethod
    def _simplify_contour(points: Polygon, max_points: int) -> Polygon:
        if len(points) <= max_points:
            return points
        indexes = np.linspace(0, len(points) - 1, max_points).round().astype(np.int32)
        return [points[int(index)] for index in indexes]

    def _interior_samples(self, contour: Polygon, image_width: int, image_height: int) -> List[Point2]:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        poly = np.asarray(contour, dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, image_width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, image_height - 1)
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
        working = mask
        if self.plane_erode_px > 0:
            k = self.plane_erode_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            eroded = cv2.erode(mask, kernel)
            if int(cv2.countNonZero(eroded)) >= self.plane_min_valid_samples * 4:
                working = eroded

        ys, xs = np.where(working > 0)
        if len(xs) < self.plane_min_valid_samples:
            raise GeometryError("top mask has too little eroded interior for depth plane")

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        span_x = max(1, x1 - x0 + 1)
        span_y = max(1, y1 - y0 + 1)
        cols = max(3, int(round(math.sqrt(self.plane_sample_count * float(span_x) / float(span_y)))))
        rows = max(3, int(math.ceil(float(self.plane_sample_count) / float(cols))))
        gx = np.linspace(x0, x1, cols)
        gy = np.linspace(y0, y1, rows)
        samples = []  # type: List[Point2]
        used = set()
        for yy in gy:
            for xx in gx:
                x = int(round(float(xx)))
                y = int(round(float(yy)))
                if 0 <= x < image_width and 0 <= y < image_height and working[y, x] > 0:
                    key = (x, y)
                    if key not in used:
                        used.add(key)
                        samples.append((float(x), float(y)))
                        if len(samples) >= self.plane_sample_count:
                            return samples

        if len(samples) < self.plane_sample_count:
            # Deterministic supplemental sampling from all interior pixels.
            step = max(1, int(len(xs) / max(1, self.plane_sample_count - len(samples))))
            for index in range(0, len(xs), step):
                key = (int(xs[index]), int(ys[index]))
                if key in used:
                    continue
                used.add(key)
                samples.append((float(key[0]), float(key[1])))
                if len(samples) >= self.plane_sample_count:
                    break
        return samples

    def classify(self, runtime_result: Mapping[str, Any]) -> ClassifiedMasks:
        width, height = self._image_size(runtime_result)
        if self.require_fixed_size and (width != self.expected_width or height != self.expected_height):
            raise GeometryError(
                "carton_bundle_grasp expects {}x{}, Runtime returned {}x{}".format(
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
            if not self._is_target(raw):
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
                    contour, self.epsilon_min, self.epsilon_max, self.epsilon_steps
                )
                ratio = float(quality.get("quad_to_contour_area_ratio", 0.0))
                if not self.min_quad_area_ratio <= ratio <= self.max_quad_area_ratio:
                    raise GeometryError("quadrilateral area ratio {:.3f} outside limits".format(ratio))
                samples = self._interior_samples(contour, width, height)
            except Exception as error:
                ignored.append({"id": source_id, "reason": "top_geometry_prepare_failed", "message": str(error)})
                continue
            accepted.append({
                "source_id": source_id,
                "class_id": _class_id(raw.get("class_id")) if _class_id(raw.get("class_id")) is not None else 0,
                "class_name": str(raw.get("class_name") or "carton_bundle_top"),
                "confidence": score,
                "bbox_xyxy": list(raw.get("bbox_xyxy") or []),
                "contour": self._simplify_contour(contour, self.contour_max_points),
                "quad": quad,
                "contour_center": _polygon_center(contour),
                "plane_sample_points": samples,
                "quadrilateral_quality": quality,
                "mask_area_px": contour_area,
            })

        if self.output_order == "largest_area":
            accepted.sort(key=lambda item: -float(item.get("mask_area_px") or 0.0))
        else:
            accepted.sort(key=lambda item: -float(item.get("confidence") or 0.0))
        return ClassifiedMasks(width, height, accepted[: self.max_targets], ignored)

    def fit_plane(self, positions: Sequence[Sequence[float]]) -> Dict[str, Any]:
        points = []  # type: List[np.ndarray]
        for raw in positions:
            point = _vec3(raw)
            if point is not None and self.min_depth_mm <= float(point[2]) <= self.max_depth_mm:
                points.append(point)
        if len(points) < self.plane_min_valid_samples:
            raise GeometryError(
                "valid plane samples {} < {}".format(len(points), self.plane_min_valid_samples)
            )
        arr = np.asarray(points, dtype=np.float64)
        rng = np.random.RandomState(41)
        best_mask = None  # type: Optional[np.ndarray]
        best_count = -1
        best_error = float("inf")
        for _ in range(self.plane_ransac_trials):
            indexes = rng.choice(arr.shape[0], 3, replace=False)
            a, b, c = arr[indexes]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, a))
            distances = np.abs(arr.dot(normal) + d)
            mask = distances <= self.plane_ransac_threshold_mm
            count = int(np.count_nonzero(mask))
            mean_error = float(distances[mask].mean()) if count else float("inf")
            if count > best_count or (count == best_count and mean_error < best_error):
                best_mask = mask
                best_count = count
                best_error = mean_error
        if best_mask is None or best_count < 3:
            raise GeometryError("RANSAC plane fitting failed")

        inliers = arr[best_mask]
        centroid = inliers.mean(axis=0)
        _u, _s, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = _unit(vh[-1], "plane normal")
        if normal[2] < 0.0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        distances = np.abs(arr.dot(normal) + d)
        final_mask = distances <= self.plane_ransac_threshold_mm
        inliers = arr[final_mask]
        if len(inliers) >= 3:
            centroid = inliers.mean(axis=0)
            _u, _s, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
            normal = _unit(vh[-1], "refined plane normal")
            if normal[2] < 0.0:
                normal = -normal
            d = -float(np.dot(normal, centroid))
        final_distances = np.abs(inliers.dot(normal) + d)
        inlier_ratio = float(len(inliers)) / float(len(arr))
        rms = math.sqrt(float(np.mean(np.square(final_distances)))) if len(inliers) else float("inf")
        if inlier_ratio < self.plane_min_inlier_ratio:
            raise GeometryError(
                "top plane inlier ratio {:.3f} < {:.3f}".format(inlier_ratio, self.plane_min_inlier_ratio)
            )
        if rms > self.plane_max_rms_mm:
            raise GeometryError("top plane RMS {:.2f}mm > {:.2f}mm".format(rms, self.plane_max_rms_mm))
        z_ref = float(np.median(inliers[:, 2]))
        tilt_deg = math.degrees(math.acos(min(1.0, max(-1.0, float(normal[2])))))
        return {
            "normal": normal,
            "d": d,
            "centroid": centroid,
            "z_ref_mm": z_ref,
            "valid_samples": int(len(arr)),
            "inlier_samples": int(len(inliers)),
            "inlier_ratio": inlier_ratio,
            "rms_mm": rms,
            "tilt_to_camera_z_deg": tilt_deg,
        }

    @staticmethod
    def intersect_corner_rays(
        ray_points: Sequence[Sequence[float]],
        plane: Mapping[str, Any],
    ) -> List[np.ndarray]:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        d = float(plane["d"])
        output = []  # type: List[np.ndarray]
        for raw in ray_points:
            ray = _vec3(raw)
            if ray is None or float(np.linalg.norm(ray)) < 1e-6:
                raise GeometryError("SDK corner ray is invalid")
            denominator = float(np.dot(normal, ray))
            if abs(denominator) < 1e-9:
                raise GeometryError("corner ray is parallel to top plane")
            scale = -d / denominator
            if scale <= 0.0 or not math.isfinite(scale):
                raise GeometryError("corner ray-plane intersection is behind camera")
            point = ray * scale
            if not np.all(np.isfinite(point)) or point[2] <= 0.0:
                raise GeometryError("corner ray-plane intersection is invalid")
            output.append(point)
        if len(output) != 4:
            raise GeometryError("exactly four corner rays are required")
        return output

    @staticmethod
    def _edge_lengths(corners: Sequence[np.ndarray]) -> List[float]:
        return [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]

    def reconstruct_rectangle(
        self,
        quad_px: Sequence[Point2],
        corners_camera: Sequence[np.ndarray],
        plane: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if len(quad_px) != 4 or len(corners_camera) != 4:
            raise GeometryError("rectangle reconstruction requires four pixels and four 3-D corners")
        p = [np.asarray(item, dtype=np.float64) for item in corners_camera]
        normal = _unit(np.asarray(plane["normal"], dtype=np.float64), "plane normal")
        edge_lengths = self._edge_lengths(p)
        pair_a = 0.5 * (edge_lengths[0] + edge_lengths[2])
        pair_b = 0.5 * (edge_lengths[1] + edge_lengths[3])

        if pair_a >= pair_b:
            observed_length = pair_a
            observed_width = pair_b
            raw_long = (p[1] - p[0]) + (p[2] - p[3])
            raw_width = (p[3] - p[0]) + (p[2] - p[1])
        else:
            observed_length = pair_b
            observed_width = pair_a
            raw_long = (p[2] - p[1]) + (p[3] - p[0])
            raw_width = (p[1] - p[0]) + (p[2] - p[3])

        raw_long = raw_long - normal * float(np.dot(raw_long, normal))
        e_length = _unit(raw_long, "bundle length axis")
        e_width = _unit(np.cross(normal, e_length), "bundle width axis")
        if float(np.dot(e_width, raw_width)) < 0.0:
            e_width = -e_width
        center = np.mean(np.asarray(p), axis=0)

        length_error = observed_length - self.length_mm
        width_error = observed_width - self.width_mm
        size_valid = (
            abs(length_error) <= self.length_tolerance_mm
            and abs(width_error) <= self.width_tolerance_mm
        )
        if not size_valid:
            raise GeometryError(
                "bundle size mismatch observed={:.1f}x{:.1f}mm expected={:.1f}x{:.1f}mm".format(
                    observed_length, observed_width, self.length_mm, self.width_mm
                )
            )

        # Generate the exact fixed-size rectangle in the recovered top plane.
        half_l = 0.5 * self.length_mm
        half_w = 0.5 * self.width_mm
        canonical = [
            center + sx * half_l * e_length + sy * half_w * e_width
            for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        ]
        # Preserve the image-corner order by assigning the nearest regularised
        # corner to each observed ray-plane corner.
        best_perm = min(
            itertools.permutations(range(4)),
            key=lambda perm: sum(float(np.linalg.norm(p[i] - canonical[perm[i]])) for i in range(4)),
        )
        regularized = [canonical[best_perm[i]] for i in range(4)]

        if self.regularize_fixed_size:
            width_mid_a = center - half_l * e_length
            width_mid_b = center + half_l * e_length
        else:
            # Observed 3-D short-edge midpoints; retained mainly for A/B testing.
            if pair_a >= pair_b:
                width_mid_a = 0.5 * (p[0] + p[3])
                width_mid_b = 0.5 * (p[1] + p[2])
            else:
                width_mid_a = 0.5 * (p[0] + p[1])
                width_mid_b = 0.5 * (p[3] + p[2])

        # A plane homography gives image coordinates for the exact 3-D midpoint
        # without doing the incorrect 2-D edge midpoint operation.
        src = np.asarray([
            [float(np.dot(point - center, e_length)), float(np.dot(point - center, e_width))]
            for point in p
        ], dtype=np.float32)
        dst = np.asarray(quad_px, dtype=np.float32)
        homography, _mask = cv2.findHomography(src, dst, method=0)
        if homography is None:
            raise GeometryError("failed to build top-plane image homography")

        def project_local(point: np.ndarray) -> List[float]:
            local = np.asarray([[[
                float(np.dot(point - center, e_length)),
                float(np.dot(point - center, e_width)),
            ]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(local, homography)[0, 0]
            return [float(projected[0]), float(projected[1])]

        regularized_px = [project_local(point) for point in regularized]
        grasp_pairs = [
            ("width_mid_a", width_mid_a, project_local(width_mid_a)),
            ("width_mid_b", width_mid_b, project_local(width_mid_b)),
        ]
        # Stable protocol order by image x, but keep explicit role in debug data.
        grasp_pairs.sort(key=lambda item: (item[2][0], item[2][1]))

        return {
            "center_camera": center,
            "raw_corners_camera": p,
            "regularized_corners_camera": regularized,
            "regularized_corners_px": regularized_px,
            "length_axis_camera": e_length,
            "width_axis_camera": e_width,
            "normal_camera": normal,
            "observed_length_mm": observed_length,
            "observed_width_mm": observed_width,
            "length_error_mm": length_error,
            "width_error_mm": width_error,
            "edge_lengths_mm": edge_lengths,
            "grasp_pairs": grasp_pairs,
        }

    def build_external_item(
        self,
        item_id: int,
        item: Mapping[str, Any],
        plane: Mapping[str, Any],
        rectangle: Mapping[str, Any],
        depth_samples: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        quad = item.get("quad") if isinstance(item.get("quad"), list) else []
        raw_corners = rectangle.get("raw_corners_camera") or []
        reg_corners = rectangle.get("regularized_corners_camera") or []
        reg_corners_px = rectangle.get("regularized_corners_px") or []
        grasp_pairs = rectangle.get("grasp_pairs") or []
        grasp_px = {}  # type: Dict[str, List[float]]
        grasp_camera = {}  # type: Dict[str, List[float]]
        grasp_roles = []  # type: List[str]
        for role, point3, point2 in grasp_pairs:
            grasp_roles.append(str(role))
            grasp_px[str(role)] = _round2(point2)
            grasp_camera[str(role)] = _round3(point3)

        plane_samples_valid = sum(1 for sample in depth_samples if bool(sample.get("depth_valid")))
        return {
            "id": int(item_id),
            "source_id": str(item.get("source_id") or ""),
            "class_id": int(item.get("class_id") or 0),
            "class_name": str(item.get("class_name") or "carton_bundle_top"),
            "confidence": round(float(item.get("confidence") or 0.0), 6),
            "geometry_mode": "FULL_TOP_FIXED_SIZE",
            "coordinate_frame": "color_camera",
            "unit": "mm",
            "contour_px": [_round2(point) for point in item.get("contour", [])],
            "observed_corners_px": {
                name: _round2(quad[index]) for index, name in enumerate(self.CORNER_NAMES)
            },
            "observed_corners_camera": {
                name: _round3(raw_corners[index]) for index, name in enumerate(self.CORNER_NAMES)
            },
            "regularized_corners_px": {
                name: _round2(reg_corners_px[index]) for index, name in enumerate(self.CORNER_NAMES)
            },
            "regularized_corners_camera": {
                name: _round3(reg_corners[index]) for index, name in enumerate(self.CORNER_NAMES)
            },
            "center_px": _round2(item.get("contour_center") or (0.0, 0.0)),
            "center_camera": _round3(rectangle["center_camera"]),
            "grasp_point_roles": grasp_roles,
            "grasp_points_px": grasp_px,
            "grasp_points_camera": grasp_camera,
            "bundle_prior": {
                "length_mm": round(self.length_mm, 3),
                "width_mm": round(self.width_mm, 3),
                "regularized_fixed_size": self.regularize_fixed_size,
            },
            "observed_size": {
                "length_mm": round(float(rectangle["observed_length_mm"]), 3),
                "width_mm": round(float(rectangle["observed_width_mm"]), 3),
                "length_error_mm": round(float(rectangle["length_error_mm"]), 3),
                "width_error_mm": round(float(rectangle["width_error_mm"]), 3),
                "edge_lengths_mm": [round(float(value), 3) for value in rectangle["edge_lengths_mm"]],
            },
            "top_plane": {
                "normal_camera": _round3(plane["normal"], 6),
                "d_mm": round(float(plane["d"]), 6),
                "centroid_camera": _round3(plane["centroid"]),
                "z_ref_mm": round(float(plane["z_ref_mm"]), 3),
                "valid_samples": int(plane["valid_samples"]),
                "inlier_samples": int(plane["inlier_samples"]),
                "inlier_ratio": round(float(plane["inlier_ratio"]), 6),
                "rms_mm": round(float(plane["rms_mm"]), 3),
                "tilt_to_camera_z_deg": round(float(plane["tilt_to_camera_z_deg"]), 3),
                "requested_samples": len(depth_samples),
                "depth_valid_samples": int(plane_samples_valid),
            },
            "axes_camera": {
                "length": _round3(rectangle["length_axis_camera"], 6),
                "width": _round3(rectangle["width_axis_camera"], 6),
                "normal": _round3(rectangle["normal_camera"], 6),
            },
            "quadrilateral_quality": dict(item.get("quadrilateral_quality") or {}),
        }
