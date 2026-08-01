"""Calibrated 3-D cardboard-box model and finger swept-volume checks.

The model is expressed in the aligned color optical frame (+X right, +Y down,
+Z forward).  A box-local frame is attached to the front-left-top inner corner:

* +X: left wall -> right wall
* +Y: top wall -> bottom wall
* +Z: front opening -> rear wall

The front plane is open. Side/top/bottom/rear constraints are applied only to
samples whose box-local Z coordinate has entered the box (Z >= 0).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass(frozen=True)
class BoxModel3D:
    origin_camera_mm: np.ndarray
    rotation_camera_from_box: np.ndarray
    inner_size_mm: np.ndarray
    safety_margin_mm: Dict[str, float]
    camera_frame_id: str
    camera_resolution: Tuple[int, int]
    intrinsics: Optional[Dict[str, float]]
    calibration: Dict[str, Any]
    source_path: Optional[str] = None

    @property
    def rotation_box_from_camera(self) -> np.ndarray:
        return self.rotation_camera_from_box.T

    def camera_to_box(self, points_camera: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera, dtype=np.float64)
        original_shape = points.shape
        points = points.reshape(-1, 3)
        result = (points - self.origin_camera_mm) @ self.rotation_camera_from_box
        return result.reshape(original_shape)

    def box_to_camera(self, points_box: np.ndarray) -> np.ndarray:
        points = np.asarray(points_box, dtype=np.float64)
        original_shape = points.shape
        points = points.reshape(-1, 3)
        result = points @ self.rotation_camera_from_box.T + self.origin_camera_mm
        return result.reshape(original_shape)

    def corners_box(self) -> np.ndarray:
        width, height, depth = self.inner_size_mm.tolist()
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [width, 0.0, 0.0],
                [width, height, 0.0],
                [0.0, height, 0.0],
                [0.0, 0.0, depth],
                [width, 0.0, depth],
                [width, height, depth],
                [0.0, height, depth],
            ],
            dtype=np.float64,
        )

    def corners_camera(self) -> np.ndarray:
        return self.box_to_camera(self.corners_box())

    def to_dict(self) -> Dict[str, Any]:
        width, height, depth = self.inner_size_mm.tolist()
        rotation = self.rotation_camera_from_box
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]
        return {
            "schema_version": "1.0",
            "message_type": "visionops_box_model_3d",
            "model_type": "calibrated_3d_cuboid",
            "coordinate_frame": self.camera_frame_id,
            "origin_definition": "front_left_top_inner_corner",
            "origin_camera_mm": [float(v) for v in self.origin_camera_mm],
            "axes_camera": {
                "x_right": [float(v) for v in x_axis],
                "y_down": [float(v) for v in y_axis],
                "z_inside": [float(v) for v in z_axis],
            },
            "rotation_camera_from_box_rows": rotation.astype(float).tolist(),
            "inner_size_mm": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth),
            },
            "safety_margin_mm": {key: float(value) for key, value in self.safety_margin_mm.items()},
            "camera_resolution": {
                "width": int(self.camera_resolution[0]),
                "height": int(self.camera_resolution[1]),
            },
            "intrinsics": self.intrinsics,
            "calibration": self.calibration,
        }


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} is a zero vector")
    return vector / norm


def _default_margins(raw: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "left": float(raw.get("left", 8.0)),
        "right": float(raw.get("right", 8.0)),
        "top": float(raw.get("top", 8.0)),
        "bottom": float(raw.get("bottom", 10.0)),
        "back": float(raw.get("back", 8.0)),
    }


def box_model_from_dict(payload: Mapping[str, Any], source_path: Optional[str] = None) -> BoxModel3D:
    if str(payload.get("model_type") or "") != "calibrated_3d_cuboid":
        raise ValueError("box model_type must be calibrated_3d_cuboid")
    origin = np.asarray(payload.get("origin_camera_mm"), dtype=np.float64).reshape(3)
    rows = payload.get("rotation_camera_from_box_rows")
    if rows is not None:
        rotation = np.asarray(rows, dtype=np.float64).reshape(3, 3)
    else:
        axes = payload.get("axes_camera") or {}
        rotation = np.column_stack(
            (
                _unit(axes.get("x_right"), "x_right"),
                _unit(axes.get("y_down"), "y_down"),
                _unit(axes.get("z_inside"), "z_inside"),
            )
        )
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if float(np.linalg.det(rotation)) < 0.0:
        rotation[:, 0] *= -1.0
    size = payload.get("inner_size_mm") or {}
    inner_size = np.asarray(
        [float(size.get("width", 0.0)), float(size.get("height", 0.0)), float(size.get("depth", 0.0))],
        dtype=np.float64,
    )
    if np.any(inner_size <= 0.0):
        raise ValueError(f"invalid box inner size: {inner_size.tolist()}")
    resolution = payload.get("camera_resolution") or {}
    camera_resolution = (int(resolution.get("width", 0)), int(resolution.get("height", 0)))
    intrinsics_raw = payload.get("intrinsics")
    intrinsics = dict(intrinsics_raw) if isinstance(intrinsics_raw, dict) else None
    return BoxModel3D(
        origin_camera_mm=origin,
        rotation_camera_from_box=rotation,
        inner_size_mm=inner_size,
        safety_margin_mm=_default_margins(payload.get("safety_margin_mm") or {}),
        camera_frame_id=str(payload.get("coordinate_frame") or "camera_color_optical_frame"),
        camera_resolution=camera_resolution,
        intrinsics=intrinsics,
        calibration=dict(payload.get("calibration") or {}),
        source_path=source_path,
    )


def load_box_model(path: Path) -> BoxModel3D:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("3-D box model root must be an object")
    return box_model_from_dict(payload, source_path=str(path))


def project_points(points_camera: np.ndarray, intrinsics: Mapping[str, float]) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    z = points[:, 2]
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    result[valid, 0] = float(intrinsics["fx"]) * points[valid, 0] / z[valid] + float(intrinsics["cx"])
    result[valid, 1] = float(intrinsics["fy"]) * points[valid, 1] / z[valid] + float(intrinsics["cy"])
    return result


def box_projection(model: BoxModel3D, intrinsics: Mapping[str, float]) -> Dict[str, Any]:
    uv = project_points(model.corners_camera(), intrinsics)
    if not np.isfinite(uv).all():
        return {"front_polygon_uv": None, "rear_polygon_uv": None, "edge_lines_uv": []}
    front = uv[[0, 1, 2, 3]]
    rear = uv[[4, 5, 6, 7]]
    edges = [[front[index].tolist(), rear[index].tolist()] for index in range(4)]
    return {
        "front_polygon_uv": front.astype(float).tolist(),
        "rear_polygon_uv": rear.astype(float).tolist(),
        "edge_lines_uv": edges,
    }


def _cross_section_corners(
    center: np.ndarray,
    closing_axis: np.ndarray,
    tangent_axis: np.ndarray,
    thickness_mm: float,
    width_mm: float,
) -> np.ndarray:
    half_t = 0.5 * float(thickness_mm)
    half_w = 0.5 * float(width_mm)
    return np.asarray(
        [
            center + closing_axis * sx * half_t + tangent_axis * sy * half_w
            for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0))
        ],
        dtype=np.float64,
    )


def sample_swept_prism(
    start_center: Sequence[float],
    end_center: Sequence[float],
    closing_axis: Sequence[float],
    tangent_axis: Sequence[float],
    thickness_mm: float,
    width_mm: float,
    sample_count: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_center, dtype=np.float64).reshape(3)
    end = np.asarray(end_center, dtype=np.float64).reshape(3)
    closing = _unit(closing_axis, "closing_axis")
    tangent = _unit(tangent_axis, "tangent_axis")
    count = max(2, int(sample_count))
    points: List[np.ndarray] = []
    sample_indices: List[int] = []
    for index, ratio in enumerate(np.linspace(0.0, 1.0, count)):
        center = start * (1.0 - ratio) + end * ratio
        corners = _cross_section_corners(center, closing, tangent, thickness_mm, width_mm)
        points.append(corners)
        sample_indices.extend([index] * len(corners))
    return np.concatenate(points, axis=0), np.asarray(sample_indices, dtype=np.int32)


def _wall_clearances(points_box: np.ndarray, size: np.ndarray) -> Dict[str, np.ndarray]:
    width, height, depth = size.tolist()
    return {
        "left": points_box[:, 0],
        "right": width - points_box[:, 0],
        "top": points_box[:, 1],
        "bottom": height - points_box[:, 1],
        "back": depth - points_box[:, 2],
    }


def check_swept_prism_against_box(
    model: BoxModel3D,
    start_center_camera: Sequence[float],
    end_center_camera: Sequence[float],
    closing_axis_camera: Sequence[float],
    tangent_axis_camera: Sequence[float],
    thickness_mm: float,
    width_mm: float,
    stage: str,
    sample_count: int = 16,
    front_entry_tolerance_mm: float = 2.0,
) -> Dict[str, Any]:
    points_camera, sample_indices = sample_swept_prism(
        start_center_camera,
        end_center_camera,
        closing_axis_camera,
        tangent_axis_camera,
        thickness_mm,
        width_mm,
        sample_count=sample_count,
    )
    points_box = model.camera_to_box(points_camera)
    active = points_box[:, 2] >= -float(front_entry_tolerance_mm)
    active_indices = np.nonzero(active)[0]
    if active_indices.size == 0:
        return {
            "stage": stage,
            "status": "outside_front",
            "active_point_count": 0,
            "minimum_clearance_mm": None,
            "nearest_wall": None,
            "physical_intersection": False,
            "safety_margin_violation": False,
        }
    q = points_box[active]
    clearances = _wall_clearances(q, model.inner_size_mm)
    margins = model.safety_margin_mm
    physical_values: List[Tuple[float, str, int]] = []
    safe_values: List[Tuple[float, str, int]] = []
    for wall, values in clearances.items():
        margin = float(margins.get(wall, 0.0))
        for local_index, value in enumerate(values.tolist()):
            physical_values.append((float(value), wall, local_index))
            safe_values.append((float(value) - margin, wall, local_index))
    physical_min, physical_wall, physical_local_index = min(physical_values, key=lambda row: row[0])
    safe_min, safe_wall, safe_local_index = min(safe_values, key=lambda row: row[0])
    physical_intersection = physical_min < 0.0
    margin_violation = safe_min < 0.0 and not physical_intersection
    status = "intersects" if physical_intersection else ("too_close" if margin_violation else "clear")
    selected_local_index = physical_local_index if physical_intersection else safe_local_index
    selected_global_index = int(active_indices[selected_local_index])
    nearest_point_camera = points_camera[selected_global_index]
    nearest_point_box = points_box[selected_global_index]
    sample_index = int(sample_indices[selected_global_index])
    return {
        "stage": stage,
        "status": status,
        "active_point_count": int(active_indices.size),
        "minimum_clearance_mm": float(physical_min),
        "minimum_safety_clearance_mm": float(safe_min),
        "nearest_wall": str(physical_wall if physical_intersection else safe_wall),
        "nearest_point_camera_mm": nearest_point_camera.astype(float).tolist(),
        "nearest_point_box_mm": nearest_point_box.astype(float).tolist(),
        "nearest_sample_index": sample_index,
        "sample_count": int(max(2, sample_count)),
        "physical_intersection": bool(physical_intersection),
        "safety_margin_violation": bool(margin_violation),
        "start_center_camera_mm": np.asarray(start_center_camera, dtype=float).tolist(),
        "end_center_camera_mm": np.asarray(end_center_camera, dtype=float).tolist(),
    }


def combine_collision_checks(checks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in checks]
    priorities = {"intersects": 3, "too_close": 2, "clear": 1, "outside_front": 0}
    worst = max(rows, key=lambda row: priorities.get(str(row.get("status")), -1)) if rows else None
    clearances = [
        float(row["minimum_clearance_mm"])
        for row in rows
        if row.get("minimum_clearance_mm") is not None
    ]
    safe_clearances = [
        float(row["minimum_safety_clearance_mm"])
        for row in rows
        if row.get("minimum_safety_clearance_mm") is not None
    ]
    return {
        "enabled": True,
        "model_type": "calibrated_3d_cuboid",
        "status": str(worst.get("status")) if worst else "unconfigured",
        "worst_stage": worst.get("stage") if worst else None,
        "nearest_wall": worst.get("nearest_wall") if worst else None,
        "minimum_clearance_mm": min(clearances) if clearances else None,
        "minimum_safety_clearance_mm": min(safe_clearances) if safe_clearances else None,
        "physical_intersection": any(bool(row.get("physical_intersection")) for row in rows),
        "safety_margin_violation": any(bool(row.get("safety_margin_violation")) for row in rows),
        "checks": rows,
    }


def validate_model_for_capture(
    model: BoxModel3D,
    image_shape: Tuple[int, int],
    intrinsics: Mapping[str, float],
    resolution_tolerance_px: int = 0,
    intrinsics_relative_tolerance: float = 0.02,
) -> Dict[str, Any]:
    height, width = image_shape
    issues: List[str] = []
    model_width, model_height = model.camera_resolution
    if model_width and model_height:
        if abs(model_width - width) > resolution_tolerance_px or abs(model_height - height) > resolution_tolerance_px:
            issues.append("camera_resolution_mismatch")
    if model.intrinsics:
        for key in ("fx", "fy", "cx", "cy"):
            expected = float(model.intrinsics.get(key, 0.0))
            current = float(intrinsics.get(key, 0.0))
            scale = max(1.0, abs(expected))
            if abs(current - expected) / scale > intrinsics_relative_tolerance:
                issues.append(f"intrinsics_{key}_mismatch")
    rotation_error = float(np.max(np.abs(model.rotation_camera_from_box.T @ model.rotation_camera_from_box - np.eye(3))))
    if rotation_error > 1e-4 or abs(float(np.linalg.det(model.rotation_camera_from_box)) - 1.0) > 1e-4:
        issues.append("rotation_not_orthonormal")
    return {
        "valid": not issues,
        "issues": issues,
        "source_path": model.source_path,
        "rotation_orthogonality_error": rotation_error,
        "inner_size_mm": {
            "width": float(model.inner_size_mm[0]),
            "height": float(model.inner_size_mm[1]),
            "depth": float(model.inner_size_mm[2]),
        },
    }
