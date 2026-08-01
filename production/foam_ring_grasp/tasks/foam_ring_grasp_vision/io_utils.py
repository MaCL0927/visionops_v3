"""Input/output helpers for M34_new.4 rim-pinch RGB-D validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import yaml  # type: ignore


@dataclass(frozen=True)
class CapturePaths:
    capture_id: str
    rgb: Path
    depth: Path
    meta: Path


def load_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("配置文件根节点必须是对象")
    return raw


def load_rgb_depth_meta(paths: CapturePaths) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rgb = cv2.imread(str(paths.rgb), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("无法读取RGB图片: %s" % paths.rgb)
    depth = cv2.imread(str(paths.depth), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("无法读取深度图片: %s" % paths.depth)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("深度图必须是uint16单通道PNG: %s, dtype=%s shape=%s" % (
            paths.depth,
            depth.dtype,
            depth.shape,
        ))
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Meta根节点必须是对象: %s" % paths.meta)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError("RGB和Depth尺寸不一致: rgb=%s depth=%s" % (rgb.shape[:2], depth.shape[:2]))
    depth_meta = meta.get("depth") or {}
    if depth_meta.get("aligned_to_color") is not True:
        raise ValueError("Meta显示Depth未对齐到RGB，不能进行逐像素几何验证")
    if depth_meta.get("calibration_ready") is not True:
        raise ValueError("Meta显示深度标定未就绪")
    return rgb, depth, meta


def resolve_intrinsics(meta: Dict[str, Any], width: int, height: int) -> Dict[str, float]:
    depth_meta = meta.get("depth") or {}
    intrinsics = depth_meta.get("intrinsics_saved") or depth_meta.get("intrinsics_source") or {}
    required = ("fx", "fy", "cx", "cy")
    missing = [name for name in required if name not in intrinsics]
    if missing:
        raise ValueError("Meta缺少深度内参: %s" % ", ".join(missing))
    result = {name: float(intrinsics[name]) for name in required}
    if result["fx"] <= 0.0 or result["fy"] <= 0.0:
        raise ValueError("深度内参fx/fy无效")
    saved_width = int(intrinsics.get("width") or width)
    saved_height = int(intrinsics.get("height") or height)
    if saved_width != width or saved_height != height:
        raise ValueError(
            "Meta内参分辨率与保存图不一致: intrinsics=%dx%d image=%dx%d"
            % (saved_width, saved_height, width, height)
        )
    return result


def resolve_capture_paths(
    data_root: Path,
    capture_id: Optional[str] = None,
    rgb_path: Optional[Path] = None,
    depth_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> CapturePaths:
    if rgb_path is not None:
        stem = rgb_path.stem
        return CapturePaths(
            capture_id=capture_id or stem,
            rgb=rgb_path,
            depth=depth_path or (data_root / "depth" / (stem + ".png")),
            meta=meta_path or (data_root / "meta" / (stem + ".json")),
        )
    if not capture_id:
        raise ValueError("必须提供capture_id或rgb路径")
    image_dir = data_root / "images"
    candidates = [image_dir / (capture_id + suffix) for suffix in (".jpg", ".jpeg", ".png")]
    rgb = next((path for path in candidates if path.exists()), candidates[0])
    return CapturePaths(
        capture_id=capture_id,
        rgb=rgb,
        depth=depth_path or (data_root / "depth" / (capture_id + ".png")),
        meta=meta_path or (data_root / "meta" / (capture_id + ".json")),
    )


def discover_captures(data_root: Path) -> List[CapturePaths]:
    image_dir = data_root / "images"
    depth_dir = data_root / "depth"
    meta_dir = data_root / "meta"
    records: List[CapturePaths] = []
    for rgb in sorted(image_dir.glob("*")):
        if rgb.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = rgb.stem
        depth = depth_dir / (stem + ".png")
        meta = meta_dir / (stem + ".json")
        if depth.exists() and meta.exists():
            records.append(CapturePaths(stem, rgb, depth, meta))
    return records


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "capture_id",
        "instance_id",
        "eligible",
        "selected",
        "initial_robot_safe_geometry",
        "warnings",
        "rejection_reasons",
        "ring_confidence",
        "mouth_confidence",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
        "tilt_deg",
        "mouth_major_mm",
        "mouth_minor_mm",
        "valid_clock_count",
        "best_clock_hour",
        "best_clock_score",
        "wall_thickness_mm",
        "target_closing_gap_mm",
        "approach_opening_mm",
        "rim_insert_depth_mm",
        "grasp_x_mm",
        "grasp_y_mm",
        "grasp_z_mm",
        "inner_containment",
        "neighbor_clearance_mm",
        "neighbor_2d_overlap_ratio",
        "neighbor_2d_clearance_mm",
        "neighbor_3d_status",
        "neighbor_3d_clearance_mm",
        "neighbor_3d_raw_minimum_clearance_mm",
        "neighbor_3d_nearest_instance_id",
        "neighbor_3d_worst_stage",
        "neighbor_3d_colliding_instance_ids",
        "full_gripper_static_status",
        "full_gripper_static_box_status",
        "full_gripper_static_neighbor_status",
        "full_gripper_static_box_clearance_mm",
        "full_gripper_static_box_safety_clearance_mm",
        "full_gripper_static_neighbor_clearance_mm",
        "full_gripper_static_box_worst_component",
        "full_gripper_static_box_nearest_wall",
        "full_gripper_static_neighbor_worst_component",
        "full_gripper_static_neighbor_nearest_instance_id",
        "full_gripper_static_finger_angle_deg",
        "full_gripper_motion_status",
        "full_gripper_motion_box_status",
        "full_gripper_motion_neighbor_status",
        "full_gripper_motion_box_clearance_mm",
        "full_gripper_motion_box_safety_clearance_mm",
        "full_gripper_motion_neighbor_clearance_mm",
        "full_gripper_motion_worst_stage",
        "full_gripper_motion_worst_sample_index",
        "full_gripper_motion_box_worst_component",
        "full_gripper_motion_box_nearest_wall",
        "full_gripper_motion_neighbor_worst_component",
        "full_gripper_motion_neighbor_nearest_instance_id",
        "full_gripper_motion_evaluated_pose_count",
        "box_wall_status",
        "box_wall_clearance_mm",
        "box_wall_safety_clearance_mm",
        "box_wall_nearest_wall",
        "box_wall_worst_stage",
        "box_wall_physical_intersection",
        "box_wall_outer_containment",
        "box_wall_inner_containment",
        "front_obstacle_status",
        "depth_valid_ratio",
        "plane_inlier_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def write_ascii_ply(path: Path, points_xyz: np.ndarray, colors_bgr: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = None
    if colors_bgr is not None:
        raw_colors = np.asarray(colors_bgr).reshape(-1, 3)[finite]
        colors = np.clip(raw_colors[:, ::-1], 0, 255).astype(np.uint8)  # BGR -> RGB
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("element vertex %d\n" % len(points))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        if colors is None:
            for point in points:
                handle.write("%.4f %.4f %.4f\n" % tuple(point.tolist()))
        else:
            for point, color in zip(points, colors):
                handle.write(
                    "%.4f %.4f %.4f %d %d %d\n"
                    % (point[0], point[1], point[2], int(color[0]), int(color[1]), int(color[2]))
                )
