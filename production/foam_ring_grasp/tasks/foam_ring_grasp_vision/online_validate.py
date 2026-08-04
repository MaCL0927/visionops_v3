"""M36.4 one-shot online RKNN + exact RGB-D geometry validation.

This command deliberately performs one complete and auditable transaction:

1. cache synchronized RGB/D2C-depth frames from Orbbec shared memory;
2. request one C++ Runtime RKNN segmentation inference;
3. retrieve only the RGB-D pair whose timestamp exactly equals
   ``capture_timestamp_ms``;
4. convert real proto polygons to :class:`SegmentationInstance` masks;
5. execute the existing ``analyze_scene`` geometry implementation;
6. save the exact inputs, raw Runtime output, geometry result and overlay.

It never substitutes a latest/nearest depth frame and never allows a
``bbox_fallback`` mask into 3-D geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

# Support both ``python -m ...online_validate`` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.common.runtime_ipc import RuntimeIpcClient  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.io_utils import (  # noqa: E402
    load_yaml,
    write_json,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.rgbd_cache import (  # noqa: E402
    RgbdCacheSettings,
    RgbdFrame,
    RgbdFrameCache,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (  # noqa: E402
    RuntimeSegmentationAdaptation,
    runtime_result_to_segmentation_instances,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.visualization import (  # noqa: E402
    depth_colormap,
    draw_overlay,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "line.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "foam_ring_online_geometry"


class OnlineGeometryError(RuntimeError):
    """A strict M36.4 safety/contract check failed."""


def _perf_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _strip_debug(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_debug(item)
            for key, item in value.items()
            if str(key) != "_debug"
        }
    if isinstance(value, list):
        return [_strip_debug(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_debug(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        raise OnlineGeometryError("box_wall.calibration_file未配置")
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise OnlineGeometryError(f"3D箱体标定文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OnlineGeometryError(f"3D箱体标定文件根节点不是对象: {path}")
    section["calibrated_model"] = payload
    section["_resolved_calibration_file"] = str(path)


def _online_settings(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    section = raw_config.get("online_geometry") or {}
    if not isinstance(section, Mapping):
        raise OnlineGeometryError("line.yaml 中 online_geometry 必须是对象")
    return {
        "enabled": bool(section.get("enabled", True)),
        "runtime_url": str(section.get("runtime_url") or "http://127.0.0.1:28081"),
        "runtime_timeout_s": max(1.0, float(section.get("runtime_timeout_s", 8.0))),
        "cache_ready_timeout_s": max(0.1, float(section.get("cache_ready_timeout_s", 3.0))),
        "require_proto_mask": bool(section.get("require_proto_mask", True)),
        "reject_bbox_fallback": bool(section.get("reject_bbox_fallback", True)),
        "minimum_mask_area_px": max(1, int(section.get("minimum_mask_area_px", 20))),
        "coordinate_space": str(section.get("coordinate_space") or "runtime_input_roi"),
        "save_exact_rgb_png": bool(section.get("save_exact_rgb_png", True)),
        "save_exact_depth_png": bool(section.get("save_exact_depth_png", True)),
        "save_depth_colormap": bool(section.get("save_depth_colormap", True)),
        "save_runtime_result": bool(section.get("save_runtime_result", True)),
        "save_overlay": bool(section.get("save_overlay", True)),
        "raw_http_enabled": bool(section.get("raw_http_enabled", True)),
        "raw_http_fallback_urllib": bool(section.get("raw_http_fallback_urllib", True)),
        "max_response_bytes": max(
            1024 * 1024,
            int(section.get("max_response_bytes", 64 * 1024 * 1024)),
        ),
    }


def _validate_frame(frame: RgbdFrame) -> None:
    if frame.rgb is None:
        raise OnlineGeometryError(
            "RGB-D缓存未保存RGB；M36.4要求online_rgbd.cache_rgb=true"
        )
    if frame.rgb.shape != (frame.height, frame.width, 3):
        raise OnlineGeometryError(
            f"缓存RGB尺寸异常: shape={frame.rgb.shape}, expected={frame.height}x{frame.width}x3"
        )
    if frame.depth_mm.shape != (frame.height, frame.width):
        raise OnlineGeometryError(
            f"缓存Depth尺寸异常: shape={frame.depth_mm.shape}, expected={frame.height}x{frame.width}"
        )
    if frame.depth_mm.dtype != np.uint16:
        raise OnlineGeometryError(f"Depth必须为uint16毫米图，当前为{frame.depth_mm.dtype}")
    if not frame.aligned_to_color:
        raise OnlineGeometryError("Depth未D2C对齐到当前RGB")
    if not frame.calibration_ready:
        raise OnlineGeometryError("Depth标定未就绪")
    if frame.fx <= 0.0 or frame.fy <= 0.0:
        raise OnlineGeometryError("RGB-D缓存内参fx/fy无效")



def _resolve_geometry_roi(
    runtime_result: Mapping[str, Any],
    frame: RgbdFrame,
    coordinate_space: str,
) -> tuple[int, int, int, int]:
    mode = str(coordinate_space or "runtime_input_roi").strip().lower()
    if mode == "full_frame":
        return (0, 0, int(frame.width), int(frame.height))
    if mode != "runtime_input_roi":
        raise OnlineGeometryError(
            "online_geometry.coordinate_space仅支持runtime_input_roi或full_frame"
        )
    document = runtime_result.get("input_roi")
    if not isinstance(document, Mapping) or not bool(document.get("enabled", False)):
        return (0, 0, int(frame.width), int(frame.height))
    pixel = document.get("pixel_xyxy") or []
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 4:
        raise OnlineGeometryError("Runtime input_roi缺少有效pixel_xyxy")
    x1, y1, x2, y2 = [int(round(float(value))) for value in pixel[:4]]
    if not (0 <= x1 < x2 <= frame.width and 0 <= y1 < y2 <= frame.height):
        raise OnlineGeometryError(
            f"Runtime input_roi越界: {(x1, y1, x2, y2)}, "
            f"frame={frame.width}x{frame.height}"
        )
    crop = document.get("crop_resolution")
    if isinstance(crop, Mapping):
        expected_width = int(crop.get("width") or 0)
        expected_height = int(crop.get("height") or 0)
        if expected_width and expected_width != x2 - x1:
            raise OnlineGeometryError(
                f"Runtime input_roi宽度契约不一致: pixel={x2-x1}, crop={expected_width}"
            )
        if expected_height and expected_height != y2 - y1:
            raise OnlineGeometryError(
                f"Runtime input_roi高度契约不一致: pixel={y2-y1}, crop={expected_height}"
            )
    return (x1, y1, x2, y2)

def _adaptation_payload(adaptation: RuntimeSegmentationAdaptation) -> Dict[str, Any]:
    per_class: Dict[str, int] = {}
    areas: list[int] = []
    for instance in adaptation.instances:
        per_class[instance.class_name] = per_class.get(instance.class_name, 0) + 1
        areas.append(instance.area_px)
    return {
        "accepted_count": int(adaptation.accepted_count),
        "rejected_count": int(adaptation.rejected_count),
        "rejected": adaptation.rejected,
        "polygon_point_count": int(adaptation.polygon_point_count),
        "class_counts": per_class,
        "mask_area_px": areas,
    }


def run_once(
    *,
    config_path: Path,
    runtime_url: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exact_match_timeout_ms: int | None = None,
    geometry_mode: str | None = None,
) -> Dict[str, Any]:
    total_started = time.perf_counter()
    config_path = config_path.expanduser().resolve()
    raw_config = load_yaml(config_path)
    if geometry_mode is not None:
        resolved_mode = str(geometry_mode).strip().lower()
        if resolved_mode not in {"first_valid", "staged", "exhaustive"}:
            raise OnlineGeometryError(
                "geometry_mode仅支持first_valid、staged或exhaustive"
            )
        optimization = raw_config.get("geometry_optimization")
        if not isinstance(optimization, dict):
            optimization = {}
            raw_config["geometry_optimization"] = optimization
        optimization["enabled"] = resolved_mode in {"first_valid", "staged"}
        optimization["mode"] = resolved_mode
    _load_box_calibration(raw_config, config_path)
    rgbd_settings = RgbdCacheSettings.from_mapping(raw_config.get("online_rgbd") or {})
    if not rgbd_settings.enabled:
        raise OnlineGeometryError("online_rgbd.enabled=false，无法执行M36.4")
    if not rgbd_settings.cache_rgb:
        raise OnlineGeometryError("M36.4要求online_rgbd.cache_rgb=true")
    online = _online_settings(raw_config)
    if not bool(online["enabled"]):
        raise OnlineGeometryError("online_geometry.enabled=false，无法执行M36.4")
    resolved_runtime_url = str(runtime_url or online["runtime_url"]).rstrip("/")
    match_timeout_ms = (
        int(exact_match_timeout_ms)
        if exact_match_timeout_ms is not None
        else int(rgbd_settings.exact_match_timeout_ms)
    )
    if match_timeout_ms < 0:
        raise OnlineGeometryError("exact_match_timeout_ms不能为负数")

    client = RuntimeIpcClient(
        resolved_runtime_url,
        float(online["runtime_timeout_s"]),
        {
            "raw_http_enabled": online["raw_http_enabled"],
            "raw_http_fallback_urllib": online["raw_http_fallback_urllib"],
            "max_response_bytes": online["max_response_bytes"],
        },
    )
    cache = RgbdFrameCache(
        rgb_name=rgbd_settings.rgb_name,
        depth_name=rgbd_settings.depth_name,
        max_frames=rgbd_settings.cache_frames,
        max_age_ms=rgbd_settings.max_age_ms,
        poll_interval_ms=rgbd_settings.poll_interval_ms,
        cache_rgb=True,
    )

    timings: Dict[str, float] = {}
    cache.start()
    try:
        ready_started = time.perf_counter()
        if not cache.wait_until_ready(float(online["cache_ready_timeout_s"])):
            raise OnlineGeometryError(
                f"RGB-D缓存未在{online['cache_ready_timeout_s']:.1f}s内就绪: {cache.status()}"
            )
        timings["cache_ready_wait_ms"] = _perf_ms(ready_started)

        status_started = time.perf_counter()
        runtime_status = client.status()
        timings["runtime_status_ms"] = _perf_ms(status_started)
        loaded_model = (
            runtime_status.get("loaded_model")
            if isinstance(runtime_status.get("loaded_model"), Mapping)
            else {}
        )
        frame_source = (
            runtime_status.get("frame_source")
            if isinstance(runtime_status.get("frame_source"), Mapping)
            else {}
        )
        if str(loaded_model.get("task_type") or "") != "segmentation":
            raise OnlineGeometryError(
                f"Runtime当前模型不是segmentation: {loaded_model.get('task_type')!r}"
            )
        if str(frame_source.get("configured_transport") or "") != "posix_shared_memory":
            raise OnlineGeometryError(
                "Runtime RGB未配置为POSIX共享内存: "
                f"{frame_source.get('configured_transport')!r}"
            )
        active_transport = str(frame_source.get("transport") or "")
        if bool(frame_source.get("fallback_active")) or active_transport == "http_jpeg_fallback":
            raise OnlineGeometryError(
                "Runtime RGB当前正在使用HTTP回退，M36.4拒绝继续: "
                f"transport={active_transport!r}, error={frame_source.get('shared_memory_last_error')!r}"
            )

        runtime_started = time.perf_counter()
        response = client.infer_once_raw()
        timings["runtime_http_ms"] = _perf_ms(runtime_started)
        timings["runtime_connect_ms"] = float(response.connect_ms)
        timings["runtime_send_ms"] = float(response.send_ms)
        timings["runtime_headers_wait_ms"] = float(response.headers_wait_ms)
        timings["runtime_body_read_ms"] = float(response.body_read_ms)

        decode_started = time.perf_counter()
        runtime_result = RuntimeIpcClient.decode_inference(response.body)
        timings["runtime_json_decode_ms"] = _perf_ms(decode_started)
        capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
        if capture_timestamp_ms <= 0:
            raise OnlineGeometryError("Runtime结果缺少有效capture_timestamp_ms")

        match_started = time.perf_counter()
        frame = cache.get_exact(
            capture_timestamp_ms,
            timeout=float(match_timeout_ms) / 1000.0,
        )
        timings["exact_rgbd_match_ms"] = _perf_ms(match_started)
        if frame is None:
            status = cache.status()
            raise OnlineGeometryError(
                "未找到与Runtime完全相同时间戳的RGB-D帧；禁止使用最新/近邻Depth: "
                f"capture_timestamp_ms={capture_timestamp_ms}, cache={status}"
            )
        _validate_frame(frame)
        timestamp_delta_ms = int(frame.timestamp_epoch_ms) - capture_timestamp_ms
        if timestamp_delta_ms != 0:
            raise OnlineGeometryError(
                f"RGB-D时间戳不是精确匹配: delta={timestamp_delta_ms}ms"
            )

        geometry_roi = _resolve_geometry_roi(
            runtime_result,
            frame,
            str(online["coordinate_space"]),
        )
        roi_x1, roi_y1, roi_x2, roi_y2 = geometry_roi
        adapt_started = time.perf_counter()
        adaptation = runtime_result_to_segmentation_instances(
            runtime_result,
            (frame.height, frame.width),
            require_proto_mask=bool(online["require_proto_mask"]),
            reject_bbox_fallback=bool(online["reject_bbox_fallback"]),
            minimum_mask_area_px=int(online["minimum_mask_area_px"]),
            geometry_roi_xyxy=geometry_roi,
        )
        timings["polygon_to_mask_ms"] = _perf_ms(adapt_started)
        if adaptation.accepted_count <= 0:
            raise OnlineGeometryError(
                "Runtime没有可用于3D几何的proto分割实例: "
                f"rejected={adaptation.rejected}"
            )

        rgb_bgr_full = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        rgb_bgr = rgb_bgr_full[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        depth_geometry = frame.depth_mm[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        intrinsics = {
            "fx": float(frame.fx),
            "fy": float(frame.fy),
            "cx": float(frame.cx) - float(roi_x1),
            "cy": float(frame.cy) - float(roi_y1),
        }
        geometry_started = time.perf_counter()
        scene = analyze_scene(
            adaptation.instances,
            depth_geometry,
            intrinsics,
            GeometryConfig(raw_config),
        )
        timings["geometry_ms"] = _perf_ms(geometry_started)

        capture_id = str(capture_timestamp_ms)
        output_dir = output_root.expanduser().resolve() / capture_id
        output_dir.mkdir(parents=True, exist_ok=True)

        save_started = time.perf_counter()
        files: Dict[str, str] = {}
        if bool(online["save_runtime_result"]):
            runtime_path = output_dir / "runtime_inference_result.json"
            runtime_path.write_bytes(response.body)
            files["runtime_result"] = str(runtime_path)
        if bool(online["save_exact_rgb_png"]):
            path = output_dir / "exact_rgb.png"
            if not cv2.imwrite(str(path), rgb_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OnlineGeometryError(f"无法保存RGB: {path}")
            files["rgb"] = str(path)
        if bool(online["save_exact_depth_png"]):
            path = output_dir / "exact_depth.png"
            if not cv2.imwrite(str(path), depth_geometry, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OnlineGeometryError(f"无法保存Depth: {path}")
            files["depth"] = str(path)
        if bool(online["save_depth_colormap"]):
            path = output_dir / "depth_colormap.jpg"
            if not cv2.imwrite(
                str(path),
                depth_colormap(depth_geometry),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise OnlineGeometryError(f"无法保存Depth可视化: {path}")
            files["depth_colormap"] = str(path)
        if bool(online["save_overlay"]):
            overlay_started = time.perf_counter()
            overlay = draw_overlay(
                rgb_bgr,
                adaptation.instances,
                scene,
                intrinsics,
            )
            timings["visualization_ms"] = _perf_ms(overlay_started)
            path = output_dir / "online_geometry_overlay.jpg"
            if not cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OnlineGeometryError(f"无法保存几何叠加图: {path}")
            files["overlay"] = str(path)
        else:
            timings["visualization_ms"] = 0.0

        timings["save_outputs_ms"] = _perf_ms(save_started)
        timings["total_ms"] = _perf_ms(total_started)
        runtime_timing = runtime_result.get("timing") if isinstance(runtime_result.get("timing"), Mapping) else {}
        scene_clean = _strip_debug(scene)
        selected = scene_clean.get("robot_candidate") if isinstance(scene_clean, Mapping) else None
        payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "message_type": "foam_ring_online_geometry_result",
            "stage": "M36.4.2_first_valid_adaptive_clock_online_geometry_once",
            "status": "ok",
            "robot_ready": False,
            "robot_ready_reason": "M36.4 is diagnostic-only; hand-eye transform, robot reachability and live trigger protocol are not enabled",
            "capture_id": capture_id,
            "capture_timestamp_ms": capture_timestamp_ms,
            "runtime": {
                "url": resolved_runtime_url,
                "result_id": runtime_result.get("result_id"),
                "frame_id": runtime_result.get("frame_id"),
                "model": runtime_result.get("model"),
                "status": {
                    "running": runtime_status.get("running"),
                    "mode": runtime_status.get("mode"),
                    "health": runtime_status.get("health"),
                    "loaded_model": loaded_model,
                    "frame_source": frame_source,
                },
                "timing": runtime_timing,
                "ipc": {
                    "transport": response.transport,
                    **client.transport_status(),
                },
            },
            "rgbd_match": {
                **frame.metadata(),
                "runtime_capture_timestamp_ms": capture_timestamp_ms,
                "matched_timestamp_ms": int(frame.timestamp_epoch_ms),
                "timestamp_delta_ms": timestamp_delta_ms,
                "exact_match_required": True,
                "nearest_fallback_allowed": False,
            },
            "segmentation_adaptation": _adaptation_payload(adaptation),
            "coordinate_space": {
                "mode": str(online["coordinate_space"]),
                "full_frame_size": {"width": int(frame.width), "height": int(frame.height)},
                "geometry_roi_xyxy": [int(value) for value in geometry_roi],
                "geometry_size": {"width": int(roi_x2 - roi_x1), "height": int(roi_y2 - roi_y1)},
                "intrinsics_transform": "cx-=roi_x1, cy-=roi_y1; fx/fy unchanged",
            },
            "intrinsics": intrinsics,
            "image": {"width": int(roi_x2 - roi_x1), "height": int(roi_y2 - roi_y1)},
            "depth": {
                "dtype": str(depth_geometry.dtype),
                "valid_count": int(np.count_nonzero(depth_geometry)),
                "total_count": int(depth_geometry.size),
                "valid_ratio": round(float(np.count_nonzero(depth_geometry)) / float(depth_geometry.size), 6),
            },
            "configuration": {
                "path": str(config_path),
                "axis_direction_enabled": bool((raw_config.get("axis_direction") or {}).get("enabled", False)),
                "full_gripper_motion_collision_enabled": bool((raw_config.get("full_gripper_motion_collision") or {}).get("enabled", False)),
                "geometry_optimization": raw_config.get("geometry_optimization") or {},
            },
            "timing_ms": {key: round(float(value), 3) for key, value in timings.items()},
            "scene": scene_clean,
            "candidate": selected,
            "files": files,
            "cache_status": cache.status(),
        }
        geometry_result_path = output_dir / "online_geometry_result.json"
        files["geometry_result"] = str(geometry_result_path)
        payload["files"] = files
        write_json(geometry_result_path, payload)
        return payload
    finally:
        cache.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M36.4.2：首个有效目标提前退出与自适应8+4钟点搜索",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-url")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exact-match-timeout-ms", type=int)
    parser.add_argument(
        "--geometry-mode",
        choices=("first_valid", "staged", "exhaustive"),
        help="覆盖line.yaml中的geometry_optimization.mode，用于first_valid/staged/exhaustive对照",
    )
    parser.add_argument(
        "--print-full-json",
        action="store_true",
        help="终端打印完整结果；默认只打印精简摘要",
    )
    return parser


def _summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else {}
    timing = payload.get("timing_ms") if isinstance(payload.get("timing_ms"), Mapping) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "capture_timestamp_ms": payload.get("capture_timestamp_ms"),
        "timestamp_delta_ms": (payload.get("rgbd_match") or {}).get("timestamp_delta_ms"),
        "rings_detected": scene.get("rings_detected"),
        "mouths_detected": scene.get("mouths_detected"),
        "matched_pairs": scene.get("matched_pairs"),
        "eligible_count": scene.get("eligible_count"),
        "selected_ring_instance_id": scene.get("selected_ring_instance_id"),
        "selected_clock_hour": scene.get("selected_clock_hour"),
        "selected_clock_angle_deg_cw_from_12": scene.get("selected_clock_angle_deg_cw_from_12"),
        "selected_clock_search_batch": scene.get("selected_clock_search_batch"),
        "runtime_total_ms": ((payload.get("runtime") or {}).get("timing") or {}).get("total_ms"),
        "polygon_to_mask_ms": timing.get("polygon_to_mask_ms"),
        "geometry_ms": timing.get("geometry_ms"),
        "geometry_mode": (scene.get("geometry_optimization") or {}).get("mode"),
        "light_candidate_count": (scene.get("geometry_optimization") or {}).get("light_candidate_count"),
        "full_candidate_evaluated_count": (scene.get("geometry_optimization") or {}).get("full_candidate_evaluated_count"),
        "full_candidate_valid_count": (scene.get("geometry_optimization") or {}).get("full_candidate_valid_count"),
        "fully_analyzed_pair_count": (scene.get("geometry_optimization") or {}).get("fully_analyzed_pair_count"),
        "deferred_pair_count": (scene.get("geometry_optimization") or {}).get("deferred_pair_count"),
        "adaptive_fallback_used": (scene.get("geometry_optimization") or {}).get("adaptive_fallback_used"),
        "early_exit_triggered": (scene.get("geometry_optimization") or {}).get("early_exit_triggered"),
        "geometry_breakdown_ms": scene.get("timing_ms"),
        "full_candidate_timing": (scene.get("timing_detail") or {}).get("full_candidates"),
        "total_ms": timing.get("total_ms"),
        "robot_ready": payload.get("robot_ready"),
        "files": payload.get("files"),
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = run_once(
            config_path=args.config,
            runtime_url=args.runtime_url,
            output_root=args.output,
            exact_match_timeout_ms=args.exact_match_timeout_ms,
            geometry_mode=args.geometry_mode,
        )
    except (OnlineGeometryError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[FAIL] M36.4.2 online geometry failed: {error}", file=sys.stderr)
        return 2
    document = payload if args.print_full_json else _summary(payload)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print("[PASS] M36.4.2 first-valid/adaptive-clock online geometry completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
