"""CLI for M35.4 directed fixed-length 3-D ring-axis rod visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from .geometry import GeometryConfig, analyze_scene, depth_pixels_to_points
from .io_utils import (
    CapturePaths,
    discover_captures,
    load_rgb_depth_meta,
    load_yaml,
    resolve_capture_paths,
    resolve_intrinsics,
    write_ascii_ply,
    write_json,
    write_summary_csv,
)
from .segmentation import (
    SegmentationInstance,
    infer_ultralytics_segmentation,
    load_yolo_segmentation_labels,
)
from .visualization import depth_colormap, draw_overlay, render_paired_axis_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "line.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M35.4：输出配对成功圆环的有向定长3D短轴杆投影",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--capture-id")
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--all", action="store_true", help="处理data-root中所有完整RGB-D组")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", type=Path, help="Ultralytics segmentation PT模型")
    parser.add_argument("--labels", type=Path, help="单帧YOLO segmentation标签")
    parser.add_argument("--labels-dir", type=Path, help="批量模式下YOLO标签目录")
    parser.add_argument("--class-names", default="foam_ring,ring_mouth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "foam_ring_rim_pinch_geometry")
    parser.add_argument("--no-scene-ply", action="store_true")
    parser.add_argument("--no-local-ply", action="store_true")
    return parser



def _load_box_calibration(raw_config: Dict[str, Any], config_path: Path) -> None:
    section = raw_config.get("box_wall")
    if not isinstance(section, dict):
        return
    if not bool(section.get("enabled", False)):
        return
    if str(section.get("model") or "") != "calibrated_3d_cuboid":
        return
    configured = section.get("calibration_file")
    if not configured:
        raise ValueError("box_wall.calibration_file未配置")
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"3D箱体标定文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"3D箱体标定文件根节点不是对象: {path}")
    section["calibrated_model"] = payload
    section["_resolved_calibration_file"] = str(path)


def _instances(
    rgb: np.ndarray,
    paths: CapturePaths,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
) -> List[SegmentationInstance]:
    names = [name.strip() for name in str(args.class_names).split(",") if name.strip()]
    label_path = args.labels
    if label_path is None and args.labels_dir:
        label_path = args.labels_dir / (paths.capture_id + ".txt")
    if label_path is not None:
        return load_yolo_segmentation_labels(label_path, rgb.shape[:2], names)
    if args.model is None:
        raise ValueError("必须提供--model或--labels/--labels-dir")
    inference = raw_config.get("inference") or {}
    device = args.device if args.device is not None else str(inference.get("device") or "auto")
    return infer_ultralytics_segmentation(
        rgb,
        args.model,
        confidence=float(inference.get("confidence", 0.20)),
        iou=float(inference.get("iou", 0.50)),
        image_size=int(inference.get("image_size", 640)),
        max_detections=int(inference.get("max_detections", 300)),
        retina_masks=bool(inference.get("retina_masks", True)),
        device=device,
    )


def _strip_debug(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_debug(item) for key, item in value.items() if key != "_debug"}
    if isinstance(value, list):
        return [_strip_debug(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _summary_rows(capture_id: str, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in scene.get("instances", []):
        center = item.get("ring_center_camera_mm") or [None, None, None]
        grasp = item.get("grasp") or {}
        best = grasp.get("best_clock_candidate") or {}
        grasp_center = best.get("grasp_center_camera_mm") or [None, None, None]
        candidates = grasp.get("clock_candidates") or []
        obstacle = best.get("local_front_obstacle") or {}
        box_wall = best.get("box_wall") or {}
        neighbor_3d = best.get("neighbor_3d") or {}
        full_static = best.get("full_gripper_static") or {}
        full_motion = best.get("full_gripper_motion") or {}
        rows.append(
            {
                "capture_id": capture_id,
                "instance_id": item.get("ring_instance_id"),
                "eligible": item.get("eligible"),
                "selected": item.get("selected"),
                "initial_robot_safe_geometry": item.get("initial_robot_safe_geometry"),
                "warnings": ";".join(item.get("warnings") or []),
                "rejection_reasons": ";".join(item.get("rejection_reasons") or []),
                "ring_confidence": item.get("ring_confidence"),
                "mouth_confidence": item.get("mouth_confidence"),
                "center_x_mm": center[0],
                "center_y_mm": center[1],
                "center_z_mm": center[2],
                "tilt_deg": item.get("tilt_deg"),
                "mouth_major_mm": item.get("mouth_major_mm"),
                "mouth_minor_mm": item.get("mouth_minor_mm"),
                "valid_clock_count": sum(1 for candidate in candidates if candidate.get("valid")),
                "best_clock_hour": best.get("clock_hour"),
                "best_clock_score": best.get("score"),
                "wall_thickness_mm": best.get("wall_thickness_mm"),
                "target_closing_gap_mm": best.get("target_closing_gap_mm"),
                "approach_opening_mm": best.get("approach_opening_mm"),
                "rim_insert_depth_mm": best.get("rim_insert_depth_mm"),
                "grasp_x_mm": grasp_center[0],
                "grasp_y_mm": grasp_center[1],
                "grasp_z_mm": grasp_center[2],
                "inner_containment": best.get("inner_finger_mouth_containment"),
                "neighbor_clearance_mm": best.get("neighbor_clearance_mm"),
                "neighbor_2d_overlap_ratio": best.get("other_ring_overlap_ratio"),
                "neighbor_2d_clearance_mm": best.get("neighbor_2d_clearance_mm"),
                "neighbor_3d_status": neighbor_3d.get("status"),
                "neighbor_3d_clearance_mm": neighbor_3d.get("minimum_clearance_mm"),
                "neighbor_3d_raw_minimum_clearance_mm": neighbor_3d.get("raw_minimum_clearance_mm"),
                "neighbor_3d_nearest_instance_id": neighbor_3d.get("nearest_instance_id"),
                "neighbor_3d_worst_stage": neighbor_3d.get("worst_stage"),
                "neighbor_3d_colliding_instance_ids": ",".join(str(value) for value in (neighbor_3d.get("colliding_instance_ids") or [])),
                "full_gripper_static_status": full_static.get("status"),
                "full_gripper_static_box_status": full_static.get("box_status"),
                "full_gripper_static_neighbor_status": full_static.get("neighbor_status"),
                "full_gripper_static_box_clearance_mm": full_static.get("box_minimum_clearance_mm"),
                "full_gripper_static_box_safety_clearance_mm": full_static.get("box_minimum_safety_clearance_mm"),
                "full_gripper_static_neighbor_clearance_mm": full_static.get("neighbor_minimum_clearance_mm"),
                "full_gripper_static_box_worst_component": full_static.get("box_worst_component"),
                "full_gripper_static_box_nearest_wall": full_static.get("box_nearest_wall"),
                "full_gripper_static_neighbor_worst_component": full_static.get("neighbor_worst_component"),
                "full_gripper_static_neighbor_nearest_instance_id": full_static.get("neighbor_nearest_instance_id"),
                "full_gripper_static_finger_angle_deg": full_static.get("finger_angle_deg"),
                "full_gripper_motion_status": full_motion.get("status"),
                "full_gripper_motion_box_status": full_motion.get("box_status"),
                "full_gripper_motion_neighbor_status": full_motion.get("neighbor_status"),
                "full_gripper_motion_box_clearance_mm": full_motion.get("box_minimum_clearance_mm"),
                "full_gripper_motion_box_safety_clearance_mm": full_motion.get("box_minimum_safety_clearance_mm"),
                "full_gripper_motion_neighbor_clearance_mm": full_motion.get("neighbor_minimum_clearance_mm"),
                "full_gripper_motion_worst_stage": full_motion.get("worst_stage"),
                "full_gripper_motion_worst_sample_index": full_motion.get("worst_sample_index"),
                "full_gripper_motion_box_worst_component": full_motion.get("box_worst_component"),
                "full_gripper_motion_box_nearest_wall": full_motion.get("box_nearest_wall"),
                "full_gripper_motion_neighbor_worst_component": full_motion.get("neighbor_worst_component"),
                "full_gripper_motion_neighbor_nearest_instance_id": full_motion.get("neighbor_nearest_instance_id"),
                "full_gripper_motion_evaluated_pose_count": full_motion.get("evaluated_pose_count"),
                "box_wall_status": box_wall.get("status"),
                "box_wall_clearance_mm": box_wall.get("clearance_mm") if box_wall.get("clearance_mm") is not None else box_wall.get("minimum_clearance_mm"),
                "box_wall_safety_clearance_mm": box_wall.get("minimum_safety_clearance_mm"),
                "box_wall_nearest_wall": box_wall.get("nearest_wall"),
                "box_wall_worst_stage": box_wall.get("worst_stage"),
                "box_wall_physical_intersection": box_wall.get("physical_intersection"),
                "box_wall_outer_containment": box_wall.get("outer_finger_containment"),
                "box_wall_inner_containment": box_wall.get("inner_finger_containment"),
                "front_obstacle_status": obstacle.get("status"),
                "depth_valid_ratio": item.get("depth_valid_ratio"),
                "plane_inlier_ratio": (item.get("plane") or {}).get("inlier_ratio"),
            }
        )
    return rows


def process_capture(
    paths: CapturePaths,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
) -> Dict[str, Any]:
    rgb, depth, meta = load_rgb_depth_meta(paths)
    intrinsics = resolve_intrinsics(meta, rgb.shape[1], rgb.shape[0])
    instances = _instances(rgb, paths, args, raw_config)
    scene = analyze_scene(instances, depth, intrinsics, GeometryConfig(raw_config))
    output_dir = args.output / paths.capture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.3",
        "message_type": "foam_ring_rim_pinch_offline_geometry_result",
        "stage": "M35.4_directed_fixed_length_3d_axis_rod_on_M35.2_geometry",
        "capture_id": paths.capture_id,
        "inputs": {
            "rgb": str(paths.rgb),
            "depth": str(paths.depth),
            "meta": str(paths.meta),
            "model": str(args.model) if args.model else None,
            "labels": str(args.labels)
            if args.labels
            else (str(args.labels_dir / (paths.capture_id + ".txt")) if args.labels_dir else None),
        },
        "image": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
        "intrinsics": intrinsics,
        "axis_direction_diagnostics": {
            "enabled": bool((raw_config.get("axis_direction") or raw_config.get("axis_visualization") or {}).get("enabled", True)),
            "mode": "directed_fixed_length_3d_axis_rod_projection",
            "note": "core pose normal remains enabled because grasp geometry depends on it",
        },
        "scene": _strip_debug(scene),
    }
    write_json(output_dir / "geometry_result.json", payload)

    output_cfg = raw_config.get("output") or {}
    robot_candidate = scene.get("robot_candidate")
    if robot_candidate and bool(output_cfg.get("save_robot_candidate", True)):
        robot_payload = {
            **robot_candidate,
            "capture_id": paths.capture_id,
            "timestamp_ms": (
                (meta.get("capture") or {}).get("timestamp_ms")
                or meta.get("timestamp_ms")
                or meta.get("timestamp_epoch_ms")
                or (meta.get("rgb") or {}).get("timestamp_epoch_ms")
            ),
        }
        write_json(output_dir / "robot_grasp_candidate.json", robot_payload)

    if bool(output_cfg.get("save_overlay", True)):
        cv2.imwrite(
            str(output_dir / "geometry_overlay.jpg"),
            draw_overlay(rgb, instances, scene, intrinsics),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    axis_cfg = raw_config.get("axis_direction") or raw_config.get("axis_visualization") or {}
    axis_diagnostics_enabled = bool(axis_cfg.get("enabled", True))
    axis_overlay_path = output_dir / "paired_axis_overlay.jpg"
    axis_json_path = output_dir / "paired_axis_projection.json"
    if axis_diagnostics_enabled and bool(output_cfg.get("save_paired_axis_overlay", True)):
        axis_overlay, axis_rows = render_paired_axis_overlay(
            rgb,
            instances,
            scene,
            intrinsics,
            axis_cfg,
        )
        cv2.imwrite(
            str(axis_overlay_path),
            axis_overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        write_json(
            axis_json_path,
            {
                "schema_version": "1.1",
                "message_type": "foam_ring_paired_axis_projection",
                "stage": "M35.4",
                "capture_id": paths.capture_id,
                "enabled": True,
                "visualization_mode": "directed_fixed_length_3d_axis_rod_projection",
                "rod_length_mm": float(axis_cfg.get("rod_length_mm", 80.0)),
                "matched_pairs": int(scene.get("matched_pairs", 0)),
                "visualized_axis_count": len(axis_rows),
                "axes": axis_rows,
            },
        )
    else:
        # Avoid leaving stale M35.4 diagnostic files when the master switch is
        # turned off and an existing output directory is reused.
        axis_overlay_path.unlink(missing_ok=True)
        axis_json_path.unlink(missing_ok=True)
    if bool(output_cfg.get("save_depth_colormap", True)):
        depth_cfg = raw_config.get("depth") or {}
        cv2.imwrite(
            str(output_dir / "depth_colormap.png"),
            depth_colormap(
                depth,
                float(depth_cfg.get("minimum_mm", 150)),
                float(depth_cfg.get("maximum_mm", 3000)),
            ),
        )

    depth_cfg = raw_config.get("depth") or {}
    minimum_depth = float(depth_cfg.get("minimum_mm", 150))
    maximum_depth = float(depth_cfg.get("maximum_mm", 3000))
    if not args.no_scene_ply and bool(output_cfg.get("save_scene_pointcloud", True)):
        scene_mask = np.ones(depth.shape, dtype=bool)
        stride = max(1, int(output_cfg.get("pointcloud_stride", 4)))
        points, pixels = depth_pixels_to_points(
            depth,
            scene_mask,
            intrinsics,
            minimum_depth,
            maximum_depth,
            stride=stride,
        )
        colors = rgb[pixels[:, 1], pixels[:, 0]] if len(pixels) else np.empty((0, 3), dtype=np.uint8)
        write_ascii_ply(output_dir / "scene_pointcloud.ply", points, colors)

    if not args.no_local_ply and bool(output_cfg.get("save_local_pointclouds", True)):
        for item in scene.get("instances", []):
            debug = item.get("_debug") or {}
            points = debug.get("plane_points")
            pixels = debug.get("plane_pixels")
            if isinstance(points, np.ndarray) and isinstance(pixels, np.ndarray) and len(points):
                colors = rgb[pixels[:, 1], pixels[:, 0]]
                write_ascii_ply(
                    output_dir / ("ring_%03d_front_pose_points.ply" % int(item.get("ring_instance_id", -1))),
                    points,
                    colors,
                )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    raw_config = load_yaml(args.config)
    _load_box_calibration(raw_config, args.config)
    if args.all:
        captures = discover_captures(args.data_root)
        if args.limit > 0:
            captures = captures[: args.limit]
    else:
        captures = [
            resolve_capture_paths(
                args.data_root,
                capture_id=args.capture_id,
                rgb_path=args.rgb,
                depth_path=args.depth,
                meta_path=args.meta,
            )
        ]
    if not captures:
        print("未发现完整RGB-D采集组", file=sys.stderr)
        return 2

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, paths in enumerate(captures, start=1):
        try:
            payload = process_capture(paths, args, raw_config)
            scene = payload["scene"]
            all_rows.extend(_summary_rows(paths.capture_id, scene))
            print(
                "[%d/%d] %s: rings=%d mouths=%d pairs=%d eligible=%d selected=%s clock=%s"
                % (
                    index,
                    len(captures),
                    paths.capture_id,
                    scene.get("rings_detected", 0),
                    scene.get("mouths_detected", 0),
                    scene.get("matched_pairs", 0),
                    scene.get("eligible_count", 0),
                    scene.get("selected_ring_instance_id"),
                    scene.get("selected_clock_hour"),
                )
            )
        except Exception as error:
            failures.append({"capture_id": paths.capture_id, "error": str(error)})
            print("[%d/%d] %s FAILED: %s" % (index, len(captures), paths.capture_id, error), file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.output / "summary.csv", all_rows)
    write_json(
        args.output / "run_summary.json",
        {
            "schema_version": "2.3",
            "message_type": "foam_ring_rim_pinch_m35_4_run_summary",
            "stage": "M35.4_directed_fixed_length_3d_axis_rod_on_M35.2_geometry",
            "processed": len(captures) - len(failures),
            "failed": len(failures),
            "candidate_rows": len(all_rows),
            "failures": failures,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
