"""Constrained 3-D box calibration from one empty-box RGB-D capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from .box_model_3d import BoxModel3D, box_projection


@dataclass
class CalibrationPlane:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    inlier_ratio: float
    residual_median_mm: float
    residual_p95_mm: float
    point_count: int


def _depth_points(
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    roi_xywh: Sequence[int],
    minimum_mm: float = 150.0,
    maximum_mm: float = 3000.0,
) -> np.ndarray:
    x, y, width, height = [int(v) for v in roi_xywh]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(depth.shape[1], x + max(1, width))
    y2 = min(depth.shape[0], y + max(1, height))
    local = depth[y1:y2, x1:x2]
    valid = (local >= minimum_mm) & (local <= maximum_mm)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    xs = xs + x1
    ys = ys + y1
    z = depth[ys, xs].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))


def _plane_from_three(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    p1, p2, p3 = points
    normal = np.cross(p2 - p1, p3 - p1)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal /= norm
    return normal, -float(np.dot(normal, p1))


def _refine_plane(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return normal, -float(np.dot(normal, centroid)), centroid


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 1200,
    inlier_threshold_mm: float = 4.0,
    random_seed: int = 3403,
) -> CalibrationPlane:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 30:
        raise ValueError(f"plane ROI has insufficient valid depth points: {len(points)}")
    rng = np.random.default_rng(int(random_seed))
    best_mask: Optional[np.ndarray] = None
    for _ in range(max(50, int(iterations))):
        indices = rng.choice(len(points), size=3, replace=False)
        candidate = _plane_from_three(points[indices])
        if candidate is None:
            continue
        normal, offset = candidate
        residuals = np.abs(points @ normal + offset)
        mask = residuals <= float(inlier_threshold_mm)
        if best_mask is None or int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < 20:
        raise ValueError("RANSAC plane fitting failed")
    normal, offset, centroid = _refine_plane(points[best_mask])
    residuals = np.abs(points @ normal + offset)
    refined_mask = residuals <= float(inlier_threshold_mm)
    inlier_residuals = residuals[refined_mask]
    return CalibrationPlane(
        normal=normal,
        offset=float(offset),
        centroid=centroid,
        inlier_ratio=float(np.mean(refined_mask)),
        residual_median_mm=float(np.percentile(inlier_residuals, 50)),
        residual_p95_mm=float(np.percentile(inlier_residuals, 95)),
        point_count=int(len(points)),
    )


def _ray(uv: Sequence[float], intrinsics: Mapping[str, float]) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    return np.asarray(
        [
            (u - float(intrinsics["cx"])) / float(intrinsics["fx"]),
            (v - float(intrinsics["cy"])) / float(intrinsics["fy"]),
            1.0,
        ],
        dtype=np.float64,
    )


def ray_plane_intersection(
    uv: Sequence[float],
    intrinsics: Mapping[str, float],
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    direction = _ray(uv, intrinsics)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-9:
        raise ValueError(f"ray is parallel to calibration plane at pixel {list(uv)}")
    scale = -float(offset) / denominator
    if scale <= 0.0:
        raise ValueError(f"ray-plane intersection is behind camera at pixel {list(uv)}")
    return direction * scale


def _orient_rear_plane(plane: CalibrationPlane) -> CalibrationPlane:
    normal = plane.normal.copy()
    offset = float(plane.offset)
    # Rear wall inward/toward-opening normal should point approximately toward camera (-Z).
    if normal[2] > 0.0:
        normal *= -1.0
        offset *= -1.0
    return CalibrationPlane(normal, offset, plane.centroid, plane.inlier_ratio, plane.residual_median_mm, plane.residual_p95_mm, plane.point_count)


def _orient_bottom_plane(plane: CalibrationPlane) -> CalibrationPlane:
    normal = plane.normal.copy()
    offset = float(plane.offset)
    # Bottom inward normal points upward, approximately -camera-Y.
    if normal[1] > 0.0:
        normal *= -1.0
        offset *= -1.0
    return CalibrationPlane(normal, offset, plane.centroid, plane.inlier_ratio, plane.residual_median_mm, plane.residual_p95_mm, plane.point_count)


def calibrate_box_model(
    depth: np.ndarray,
    intrinsics: Mapping[str, float],
    rear_roi_xywh: Sequence[int],
    bottom_roi_xywh: Sequence[int],
    rear_corners_uv: Sequence[Sequence[float]],
    front_bottom_edge_uv: Sequence[Sequence[float]],
    camera_frame_id: str = "camera_color_optical_frame",
    safety_margin_mm: Optional[Mapping[str, float]] = None,
    source: Optional[Mapping[str, Any]] = None,
    ransac_iterations: int = 1200,
    inlier_threshold_mm: float = 4.0,
) -> BoxModel3D:
    if len(rear_corners_uv) != 4:
        raise ValueError("rear_corners_uv must contain TL, TR, BR, BL")
    if len(front_bottom_edge_uv) != 2:
        raise ValueError("front_bottom_edge_uv must contain left and right points")
    rear_points = _depth_points(depth, intrinsics, rear_roi_xywh)
    bottom_points = _depth_points(depth, intrinsics, bottom_roi_xywh)
    rear_plane = _orient_rear_plane(
        fit_plane_ransac(rear_points, ransac_iterations, inlier_threshold_mm, random_seed=34031)
    )
    bottom_plane = _orient_bottom_plane(
        fit_plane_ransac(bottom_points, ransac_iterations, inlier_threshold_mm, random_seed=34032)
    )

    z_axis = -rear_plane.normal
    z_axis /= np.linalg.norm(z_axis)
    y_raw = -bottom_plane.normal
    y_axis = y_raw - z_axis * float(np.dot(y_raw, z_axis))
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    if x_axis[0] < 0.0:
        x_axis *= -1.0
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    back_camera = np.asarray(
        [ray_plane_intersection(uv, intrinsics, rear_plane.normal, rear_plane.offset) for uv in rear_corners_uv],
        dtype=np.float64,
    )
    front_bottom_camera = np.asarray(
        [ray_plane_intersection(uv, intrinsics, bottom_plane.normal, bottom_plane.offset) for uv in front_bottom_edge_uv],
        dtype=np.float64,
    )
    back_box_unshifted = back_camera @ rotation
    front_box_unshifted = front_bottom_camera @ rotation

    # Ordered rear corners: TL, TR, BR, BL. The front bottom edge contributes a
    # same-depth estimate for left/right and a robust front-Z estimate.
    x_left = float(np.median([back_box_unshifted[0, 0], back_box_unshifted[3, 0], front_box_unshifted[0, 0]]))
    x_right = float(np.median([back_box_unshifted[1, 0], back_box_unshifted[2, 0], front_box_unshifted[1, 0]]))
    y_top = float(np.mean(back_box_unshifted[[0, 1], 1]))
    y_bottom = float(np.median([back_box_unshifted[2, 1], back_box_unshifted[3, 1], *front_box_unshifted[:, 1].tolist()]))
    z_front = float(np.mean(front_box_unshifted[:, 2]))
    z_rear = float(np.mean(back_box_unshifted[:, 2]))
    inner_size = np.asarray([x_right - x_left, y_bottom - y_top, z_rear - z_front], dtype=np.float64)
    if np.any(inner_size <= 50.0):
        raise ValueError(f"calibrated box dimensions are implausible: {inner_size.tolist()}")
    origin_camera = rotation @ np.asarray([x_left, y_top, z_front], dtype=np.float64)

    orthogonality_raw_deg = math.degrees(
        math.acos(float(np.clip(abs(np.dot(rear_plane.normal, bottom_plane.normal)), 0.0, 1.0)))
    )
    calibration = {
        "status": "calibrated_requires_visual_confirmation",
        "method": "rear_and_bottom_ransac_plus_orthogonal_cuboid_constraints",
        "source": dict(source or {}),
        "input": {
            "rear_roi_xywh": [int(v) for v in rear_roi_xywh],
            "bottom_roi_xywh": [int(v) for v in bottom_roi_xywh],
            "rear_corners_uv_tl_tr_br_bl": [[float(v) for v in uv] for uv in rear_corners_uv],
            "front_bottom_edge_uv_left_right": [[float(v) for v in uv] for uv in front_bottom_edge_uv],
        },
        "rear_plane": {
            "normal_toward_opening": rear_plane.normal.astype(float).tolist(),
            "offset": float(rear_plane.offset),
            "inlier_ratio": rear_plane.inlier_ratio,
            "residual_median_mm": rear_plane.residual_median_mm,
            "residual_p95_mm": rear_plane.residual_p95_mm,
            "point_count": rear_plane.point_count,
        },
        "bottom_plane": {
            "normal_inward": bottom_plane.normal.astype(float).tolist(),
            "offset": float(bottom_plane.offset),
            "inlier_ratio": bottom_plane.inlier_ratio,
            "residual_median_mm": bottom_plane.residual_median_mm,
            "residual_p95_mm": bottom_plane.residual_p95_mm,
            "point_count": bottom_plane.point_count,
        },
        "raw_rear_bottom_normal_angle_deg": float(orthogonality_raw_deg),
    }
    margins = {
        "left": 8.0,
        "right": 8.0,
        "top": 8.0,
        "bottom": 10.0,
        "back": 8.0,
    }
    if safety_margin_mm:
        margins.update({key: float(value) for key, value in safety_margin_mm.items()})
    return BoxModel3D(
        origin_camera_mm=origin_camera,
        rotation_camera_from_box=rotation,
        inner_size_mm=inner_size,
        safety_margin_mm=margins,
        camera_frame_id=camera_frame_id,
        camera_resolution=(int(depth.shape[1]), int(depth.shape[0])),
        intrinsics={key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")},
        calibration=calibration,
    )


def draw_calibration_overlay(
    rgb_bgr: np.ndarray,
    model: BoxModel3D,
    intrinsics: Mapping[str, float],
    rear_roi_xywh: Optional[Sequence[int]] = None,
    bottom_roi_xywh: Optional[Sequence[int]] = None,
    rear_corners_uv: Optional[Sequence[Sequence[float]]] = None,
    front_bottom_edge_uv: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    output = rgb_bgr.copy()
    projection = box_projection(model, intrinsics)
    front = projection.get("front_polygon_uv")
    rear = projection.get("rear_polygon_uv")
    if front:
        cv2.polylines(output, [np.rint(np.asarray(front)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2, cv2.LINE_AA)
    if rear:
        cv2.polylines(output, [np.rint(np.asarray(rear)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 220, 0), 2, cv2.LINE_AA)
    for edge in projection.get("edge_lines_uv") or []:
        p1 = tuple(np.rint(np.asarray(edge[0])).astype(int).tolist())
        p2 = tuple(np.rint(np.asarray(edge[1])).astype(int).tolist())
        cv2.line(output, p1, p2, (255, 220, 0), 1, cv2.LINE_AA)
    for roi, color in ((rear_roi_xywh, (0, 180, 0)), (bottom_roi_xywh, (0, 180, 255))):
        if roi is not None:
            x, y, width, height = [int(v) for v in roi]
            cv2.rectangle(output, (x, y), (x + width, y + height), color, 1, cv2.LINE_AA)
    for points, color in ((rear_corners_uv, (0, 255, 0)), (front_bottom_edge_uv, (0, 0, 255))):
        if points:
            for index, uv in enumerate(points):
                point = (int(round(float(uv[0]))), int(round(float(uv[1]))))
                cv2.circle(output, point, 4, color, -1, cv2.LINE_AA)
                cv2.putText(output, str(index + 1), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    size = model.inner_size_mm
    cv2.putText(
        output,
        "3D BOX W=%.1f H=%.1f D=%.1f mm" % (size[0], size[1], size[2]),
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output
