"""Reusable exact-RGB-D online geometry processor for M36.4/M36.5.

The one-shot validator and the persistent trigger service share this class.  It
owns one long-lived Runtime IPC client and one long-lived synchronized RGB-D
cache, while every trigger remains a strict transaction:

* request one Runtime segmentation result;
* match only the exact ``capture_timestamp_ms`` RGB-D pair;
* reject HTTP RGB fallback and bbox-only masks;
* run the configured foam-ring geometry;
* optionally generate an in-memory overlay and/or save a complete debug bundle.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.common.runtime_ipc import RuntimeIpcClient  # noqa: E402
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.geometry import (  # noqa: E402
    GeometryConfig,
    analyze_scene,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.hybrid_grasp import (  # noqa: E402
    HybridGraspConfig,
    run_hybrid_grasp,
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
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.side_ring_offline_validate import (  # noqa: E402
    draw_side_ring_fit_overlay,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "line.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "foam_ring_online_geometry"


class OnlineGeometryError(RuntimeError):
    """A strict online geometry safety/contract check failed."""


@dataclass(frozen=True)
class OnlineProcessResult:
    payload: Dict[str, Any]
    overlay_jpeg: bytes | None


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


def _apply_geometry_mode(raw_config: Dict[str, Any], geometry_mode: str | None) -> None:
    if geometry_mode is None:
        return
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


def _validate_frame(frame: RgbdFrame) -> None:
    if frame.rgb is None:
        raise OnlineGeometryError(
            "RGB-D缓存未保存RGB；在线几何要求online_rgbd.cache_rgb=true"
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



@dataclass
class PreparedOnlineGeometry:
    """Inference/RGB-D adaptation output waiting for serialized geometry."""

    request_id: str | None
    started_monotonic: float
    prepared_monotonic: float
    timings: Dict[str, float]
    response_body: bytes
    runtime_result: Dict[str, Any]
    runtime_status: Dict[str, Any]
    loaded_model: Mapping[str, Any]
    frame_source: Mapping[str, Any]
    runtime_ipc: Dict[str, Any]
    frame_metadata: Dict[str, Any]
    timestamp_delta_ms: int
    adaptation: RuntimeSegmentationAdaptation
    geometry_roi: tuple[int, int, int, int]
    rgb_bgr: np.ndarray
    depth_geometry: np.ndarray
    intrinsics: Dict[str, float]
    capture_timestamp_ms: int


class OnlineGeometryProcessor:
    """Long-lived Runtime client + exact RGB-D cache + geometry configuration.

    M36.5 may run :meth:`prepare` and :meth:`finish` on separate workers. The
    exact RGB-D pixels and rasterized masks are copied into
    :class:`PreparedOnlineGeometry` before the inference worker proceeds to the
    next request, so later cache eviction cannot invalidate a queued explicit
    trigger.
    """

    def __init__(
        self,
        *,
        config_path: Path = DEFAULT_CONFIG,
        runtime_url: str | None = None,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        exact_match_timeout_ms: int | None = None,
        geometry_mode: str | None = None,
        runtime_status_ttl_ms: int = 0,
        overlay_jpeg_quality: int = 92,
        client_factory: Callable[..., Any] = RuntimeIpcClient,
        cache_factory: Callable[..., Any] = RgbdFrameCache,
        analyze_fn: Callable[..., Dict[str, Any]] = analyze_scene,
    ) -> None:
        self.config_path = config_path.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.raw_config = load_yaml(self.config_path)
        _apply_geometry_mode(self.raw_config, geometry_mode)
        _load_box_calibration(self.raw_config, self.config_path)
        self.rgbd_settings = RgbdCacheSettings.from_mapping(
            self.raw_config.get("online_rgbd") or {}
        )
        if not self.rgbd_settings.enabled:
            raise OnlineGeometryError("online_rgbd.enabled=false，无法执行在线几何")
        if not self.rgbd_settings.cache_rgb:
            raise OnlineGeometryError("在线几何要求online_rgbd.cache_rgb=true")
        self.online = _online_settings(self.raw_config)
        if not bool(self.online["enabled"]):
            raise OnlineGeometryError("online_geometry.enabled=false，无法执行在线几何")
        self.runtime_url = str(runtime_url or self.online["runtime_url"]).rstrip("/")
        self.match_timeout_ms = (
            int(exact_match_timeout_ms)
            if exact_match_timeout_ms is not None
            else int(self.rgbd_settings.exact_match_timeout_ms)
        )
        if self.match_timeout_ms < 0:
            raise OnlineGeometryError("exact_match_timeout_ms不能为负数")
        self.runtime_status_ttl_ms = max(0, int(runtime_status_ttl_ms))
        self.overlay_jpeg_quality = min(100, max(40, int(overlay_jpeg_quality)))
        self.client = client_factory(
            self.runtime_url,
            float(self.online["runtime_timeout_s"]),
            {
                "raw_http_enabled": self.online["raw_http_enabled"],
                "raw_http_fallback_urllib": self.online["raw_http_fallback_urllib"],
                "max_response_bytes": self.online["max_response_bytes"],
            },
        )
        self.cache = cache_factory(
            rgb_name=self.rgbd_settings.rgb_name,
            depth_name=self.rgbd_settings.depth_name,
            max_frames=self.rgbd_settings.cache_frames,
            max_age_ms=self.rgbd_settings.max_age_ms,
            poll_interval_ms=self.rgbd_settings.poll_interval_ms,
            cache_rgb=True,
        )
        self.geometry_config = GeometryConfig(self.raw_config)
        self.hybrid_config = HybridGraspConfig.from_mapping(self.raw_config)
        self._analyze_fn = analyze_fn
        self._started = False
        self._lifecycle_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._geometry_lock = threading.Lock()
        self._runtime_status_lock = threading.Lock()
        self._runtime_status: Dict[str, Any] | None = None
        self._runtime_status_monotonic = 0.0
        self._runtime_status_error: str | None = None
        self._cache_ready_wait_ms = 0.0

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self.cache.start()
            ready_started = time.perf_counter()
            if not self.cache.wait_until_ready(float(self.online["cache_ready_timeout_s"])):
                status = self.cache.status()
                self.cache.stop()
                raise OnlineGeometryError(
                    f"RGB-D缓存未在{self.online['cache_ready_timeout_s']:.1f}s内就绪: {status}"
                )
            self._cache_ready_wait_ms = _perf_ms(ready_started)
            self._started = True

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._started:
                return
            self.cache.stop()
            self._started = False

    def runtime_status(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        with self._runtime_status_lock:
            if (
                not force
                and self._runtime_status is not None
                and self.runtime_status_ttl_ms > 0
                and (now - self._runtime_status_monotonic) * 1000.0
                <= self.runtime_status_ttl_ms
            ):
                return dict(self._runtime_status)
        try:
            document = self.client.status()
        except Exception as error:
            with self._runtime_status_lock:
                self._runtime_status_error = str(error)
            raise
        with self._runtime_status_lock:
            self._runtime_status = dict(document)
            self._runtime_status_monotonic = time.monotonic()
            self._runtime_status_error = None
            return dict(document)

    @staticmethod
    def _validate_runtime_status(
        runtime_status: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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
                "Runtime RGB当前正在使用HTTP回退，在线3D拒绝继续: "
                f"transport={active_transport!r}, "
                f"error={frame_source.get('shared_memory_last_error')!r}"
            )
        return loaded_model, frame_source

    def status(self, *, refresh_runtime: bool = False) -> Dict[str, Any]:
        runtime: Dict[str, Any] | None = None
        runtime_error: str | None = None
        try:
            runtime = self.runtime_status(force=refresh_runtime)
        except Exception as error:
            runtime_error = str(error)
        return {
            "started": bool(self._started),
            "runtime_url": self.runtime_url,
            "runtime": runtime,
            "runtime_error": runtime_error or self._runtime_status_error,
            "runtime_ipc": self.client.transport_status(),
            "cache": self.cache.status(),
            "configuration": {
                "path": str(self.config_path),
                "geometry_mode": str(
                    (self.raw_config.get("geometry_optimization") or {}).get("mode")
                    or "exhaustive"
                ),
                "exact_match_timeout_ms": int(self.match_timeout_ms),
                "hybrid_grasp_enabled": bool(self.hybrid_config.enabled),
                "branch_priority": [
                    "m36_mouth_visible_rim_pinch",
                    "m37_side_ring_near_visible_crown",
                ],
            },
        }

    def prepare(self, *, request_id: str | None = None) -> PreparedOnlineGeometry:
        """Run Runtime inference and freeze the exact RGB-D/mask transaction."""
        if not self._started:
            raise OnlineGeometryError("OnlineGeometryProcessor尚未启动")
        if not self._inference_lock.acquire(blocking=False):
            raise OnlineGeometryError("Runtime推理准备线程正忙")
        try:
            total_started = time.monotonic()
            timings: Dict[str, float] = {"cache_ready_wait_ms": 0.0}
            status_started = time.perf_counter()
            runtime_status = self.runtime_status(force=False)
            timings["runtime_status_ms"] = _perf_ms(status_started)
            loaded_model, frame_source = self._validate_runtime_status(runtime_status)

            runtime_started = time.perf_counter()
            response = self.client.infer_once_raw()
            timings["runtime_http_ms"] = _perf_ms(runtime_started)
            timings["runtime_connect_ms"] = float(response.connect_ms)
            timings["runtime_send_ms"] = float(response.send_ms)
            timings["runtime_headers_wait_ms"] = float(response.headers_wait_ms)
            timings["runtime_body_read_ms"] = float(response.body_read_ms)

            decode_started = time.perf_counter()
            runtime_result = self.client.decode_inference(response.body)
            timings["runtime_json_decode_ms"] = _perf_ms(decode_started)
            capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
            if capture_timestamp_ms <= 0:
                raise OnlineGeometryError("Runtime结果缺少有效capture_timestamp_ms")

            match_started = time.perf_counter()
            frame = self.cache.get_exact(
                capture_timestamp_ms,
                timeout=float(self.match_timeout_ms) / 1000.0,
            )
            timings["exact_rgbd_match_ms"] = _perf_ms(match_started)
            if frame is None:
                raise OnlineGeometryError(
                    "未找到与Runtime完全相同时间戳的RGB-D帧；禁止使用最新/近邻Depth: "
                    f"capture_timestamp_ms={capture_timestamp_ms}, cache={self.cache.status()}"
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
                str(self.online["coordinate_space"]),
            )
            roi_x1, roi_y1, roi_x2, roi_y2 = geometry_roi
            adapt_started = time.perf_counter()
            adaptation = runtime_result_to_segmentation_instances(
                runtime_result,
                (frame.height, frame.width),
                require_proto_mask=bool(self.online["require_proto_mask"]),
                reject_bbox_fallback=bool(self.online["reject_bbox_fallback"]),
                minimum_mask_area_px=int(self.online["minimum_mask_area_px"]),
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
            timings["prepare_total_ms"] = (time.monotonic() - total_started) * 1000.0
            return PreparedOnlineGeometry(
                request_id=request_id,
                started_monotonic=total_started,
                prepared_monotonic=time.monotonic(),
                timings=timings,
                response_body=bytes(response.body),
                runtime_result=runtime_result,
                runtime_status=dict(runtime_status),
                loaded_model=loaded_model,
                frame_source=frame_source,
                runtime_ipc={
                    "transport": response.transport,
                    **self.client.transport_status(),
                },
                frame_metadata=frame.metadata(),
                timestamp_delta_ms=timestamp_delta_ms,
                adaptation=adaptation,
                geometry_roi=geometry_roi,
                rgb_bgr=rgb_bgr,
                depth_geometry=depth_geometry,
                intrinsics=intrinsics,
                capture_timestamp_ms=capture_timestamp_ms,
            )
        finally:
            self._inference_lock.release()

    def finish(
        self,
        prepared: PreparedOnlineGeometry,
        *,
        save_debug: bool | None = None,
        generate_overlay: bool = False,
        stage: str = "M36.5_persistent_trigger_service",
        geometry_queue_wait_ms: float | None = None,
    ) -> OnlineProcessResult:
        """Run serialized 3-D geometry and optional evidence generation."""
        if not self._started:
            raise OnlineGeometryError("OnlineGeometryProcessor尚未启动")
        if not self._geometry_lock.acquire(blocking=False):
            raise OnlineGeometryError("在线三维几何线程正忙")
        try:
            return self._finish_locked(
                prepared,
                save_debug=save_debug,
                generate_overlay=generate_overlay,
                stage=stage,
                geometry_queue_wait_ms=geometry_queue_wait_ms,
            )
        finally:
            self._geometry_lock.release()

    def process(
        self,
        *,
        request_id: str | None = None,
        save_debug: bool | None = None,
        generate_overlay: bool = False,
        stage: str = "M36.5_persistent_trigger_service",
    ) -> OnlineProcessResult:
        prepared = self.prepare(request_id=request_id)
        return self.finish(
            prepared,
            save_debug=save_debug,
            generate_overlay=generate_overlay,
            stage=stage,
            geometry_queue_wait_ms=0.0,
        )

    def _finish_locked(
        self,
        prepared: PreparedOnlineGeometry,
        *,
        save_debug: bool | None,
        generate_overlay: bool,
        stage: str,
        geometry_queue_wait_ms: float | None,
    ) -> OnlineProcessResult:
        timings = dict(prepared.timings)
        if geometry_queue_wait_ms is None:
            geometry_queue_wait_ms = (
                time.monotonic() - prepared.prepared_monotonic
            ) * 1000.0
        timings["geometry_queue_wait_ms"] = max(0.0, float(geometry_queue_wait_ms))

        geometry_started = time.perf_counter()
        if self.hybrid_config.enabled:
            scene = run_hybrid_grasp(
                prepared.adaptation.instances,
                prepared.depth_geometry,
                prepared.intrinsics,
                raw_config=self.raw_config,
                geometry_config=self.geometry_config,
                analyze_fn=self._analyze_fn,
            )
        else:
            scene = self._analyze_fn(
                prepared.adaptation.instances,
                prepared.depth_geometry,
                prepared.intrinsics,
                self.geometry_config,
            )
        timings["geometry_ms"] = _perf_ms(geometry_started)
        scene_clean = _strip_debug(scene)
        selected = scene_clean.get("robot_candidate") if isinstance(scene_clean, Mapping) else None

        if save_debug is None:
            save_flags = {
                "runtime": bool(self.online["save_runtime_result"]),
                "rgb": bool(self.online["save_exact_rgb_png"]),
                "depth": bool(self.online["save_exact_depth_png"]),
                "depth_colormap": bool(self.online["save_depth_colormap"]),
                "overlay": bool(self.online["save_overlay"]),
                "geometry_result": True,
            }
        elif save_debug:
            save_flags = {
                "runtime": True,
                "rgb": True,
                "depth": True,
                "depth_colormap": True,
                "overlay": True,
                "geometry_result": True,
            }
        else:
            save_flags = {
                "runtime": False,
                "rgb": False,
                "depth": False,
                "depth_colormap": False,
                "overlay": False,
                "geometry_result": False,
            }

        overlay_jpeg: bytes | None = None
        overlay_needed = bool(generate_overlay or save_flags["overlay"])
        if overlay_needed:
            overlay_started = time.perf_counter()
            overlay = draw_overlay(
                prepared.rgb_bgr,
                prepared.adaptation.instances,
                scene,
                prepared.intrinsics,
            )
            layering = (
                scene.get("depth_layering")
                if isinstance(scene, Mapping)
                and isinstance(scene.get("depth_layering"), Mapping)
                else {}
            )
            depth_rows = {
                int(row.get("ring_instance_id")): row
                for row in (layering.get("candidates") or [])
                if isinstance(row, Mapping)
                and row.get("ring_instance_id") is not None
            }
            selected_depth_id = scene.get("selected_ring_instance_id") if isinstance(scene, Mapping) else None
            for instance in prepared.adaptation.instances:
                if instance.class_name != "foam_ring":
                    continue
                row = depth_rows.get(int(instance.instance_id))
                if not row:
                    continue
                x1, y1, _x2, _y2 = [int(value) for value in instance.bbox_xyxy]
                layer_value = row.get("depth_layer_index")
                depth_value = row.get("surface_depth_mm")
                label = "L{} z={}".format(
                    layer_value if layer_value is not None else "?",
                    f"{float(depth_value):.0f}" if depth_value is not None else "?",
                )
                color = (0, 255, 255) if selected_depth_id is not None and int(instance.instance_id) == int(selected_depth_id) else ((0, 220, 0) if layer_value == 0 else (0, 165, 255))
                cv2.putText(
                    overlay,
                    label,
                    (max(2, x1), max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            side_branch = (
                scene.get("side_ring_branch")
                if isinstance(scene, Mapping)
                and isinstance(scene.get("side_ring_branch"), Mapping)
                else {}
            )
            side_fits = side_branch.get("fits") if isinstance(side_branch, Mapping) else []
            selected_side_id = (
                side_branch.get("selected_ring_instance_id")
                if isinstance(side_branch, Mapping)
                else None
            )
            if side_fits:
                overlay = draw_side_ring_fit_overlay(
                    overlay,
                    prepared.adaptation.instances,
                    side_fits,
                    prepared.intrinsics,
                    int(selected_side_id) if selected_side_id is not None else None,
                )
            branch_label = str(
                scene.get("selected_grasp_branch")
                if isinstance(scene, Mapping)
                else "none"
            )
            cv2.putText(
                overlay,
                "M38.5 hybrid branch: " + branch_label,
                (10, max(36, overlay.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                overlay,
                [cv2.IMWRITE_JPEG_QUALITY, self.overlay_jpeg_quality],
            )
            if not ok:
                raise OnlineGeometryError("无法编码在线几何叠加图")
            overlay_jpeg = encoded.tobytes()
            timings["visualization_ms"] = _perf_ms(overlay_started)
        else:
            timings["visualization_ms"] = 0.0

        capture_id = str(prepared.capture_timestamp_ms)
        files: Dict[str, str] = {}
        output_dir: Path | None = None
        save_started = time.perf_counter()
        if any(save_flags.values()):
            output_dir = self.output_root / capture_id
            output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is not None and save_flags["runtime"]:
            path = output_dir / "runtime_inference_result.json"
            path.write_bytes(prepared.response_body)
            files["runtime_result"] = str(path)
        if output_dir is not None and save_flags["rgb"]:
            path = output_dir / "exact_rgb.png"
            if not cv2.imwrite(
                str(path), prepared.rgb_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            ):
                raise OnlineGeometryError(f"无法保存RGB: {path}")
            files["rgb"] = str(path)
        if output_dir is not None and save_flags["depth"]:
            path = output_dir / "exact_depth.png"
            if not cv2.imwrite(
                str(path), prepared.depth_geometry, [cv2.IMWRITE_PNG_COMPRESSION, 1]
            ):
                raise OnlineGeometryError(f"无法保存Depth: {path}")
            files["depth"] = str(path)
        if output_dir is not None and save_flags["depth_colormap"]:
            path = output_dir / "depth_colormap.jpg"
            if not cv2.imwrite(
                str(path),
                depth_colormap(prepared.depth_geometry),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise OnlineGeometryError(f"无法保存Depth可视化: {path}")
            files["depth_colormap"] = str(path)
        if output_dir is not None and save_flags["overlay"] and overlay_jpeg is not None:
            path = output_dir / "online_geometry_overlay.jpg"
            path.write_bytes(overlay_jpeg)
            files["overlay"] = str(path)
        timings["save_outputs_ms"] = _perf_ms(save_started)
        timings["total_ms"] = (
            time.monotonic() - prepared.started_monotonic
        ) * 1000.0

        runtime_result = prepared.runtime_result
        runtime_timing = (
            runtime_result.get("timing")
            if isinstance(runtime_result.get("timing"), Mapping)
            else {}
        )
        roi_x1, roi_y1, roi_x2, roi_y2 = prepared.geometry_roi
        valid_count = int(np.count_nonzero(prepared.depth_geometry))
        frame_width = int(prepared.frame_metadata.get("width") or 0)
        frame_height = int(prepared.frame_metadata.get("height") or 0)
        payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "message_type": "foam_ring_online_geometry_result",
            "stage": str(stage),
            "status": "ok",
            "request_id": prepared.request_id,
            "robot_ready": False,
            "robot_ready_reason": (
                "M38.5 returns camera-frame candidate/rejection data only; hand-eye transform, "
                "robot reachability and final robot protocol are not enabled"
            ),
            "capture_id": capture_id,
            "capture_timestamp_ms": prepared.capture_timestamp_ms,
            "runtime": {
                "url": self.runtime_url,
                "result_id": runtime_result.get("result_id"),
                "frame_id": runtime_result.get("frame_id"),
                "model": runtime_result.get("model"),
                "status": {
                    "running": prepared.runtime_status.get("running"),
                    "mode": prepared.runtime_status.get("mode"),
                    "health": prepared.runtime_status.get("health"),
                    "loaded_model": prepared.loaded_model,
                    "frame_source": prepared.frame_source,
                },
                "timing": runtime_timing,
                "ipc": prepared.runtime_ipc,
            },
            "rgbd_match": {
                **prepared.frame_metadata,
                "runtime_capture_timestamp_ms": prepared.capture_timestamp_ms,
                "matched_timestamp_ms": int(
                    prepared.frame_metadata.get("timestamp_epoch_ms") or 0
                ),
                "timestamp_delta_ms": prepared.timestamp_delta_ms,
                "exact_match_required": True,
                "nearest_fallback_allowed": False,
            },
            "segmentation_adaptation": _adaptation_payload(prepared.adaptation),
            "coordinate_space": {
                "mode": str(self.online["coordinate_space"]),
                "full_frame_size": {
                    "width": frame_width,
                    "height": frame_height,
                },
                "geometry_roi_xyxy": [int(value) for value in prepared.geometry_roi],
                "geometry_size": {
                    "width": int(roi_x2 - roi_x1),
                    "height": int(roi_y2 - roi_y1),
                },
                "intrinsics_transform": "cx-=roi_x1, cy-=roi_y1; fx/fy unchanged",
            },
            "intrinsics": prepared.intrinsics,
            "image": {
                "width": int(roi_x2 - roi_x1),
                "height": int(roi_y2 - roi_y1),
            },
            "depth": {
                "dtype": str(prepared.depth_geometry.dtype),
                "valid_count": valid_count,
                "total_count": int(prepared.depth_geometry.size),
                "valid_ratio": round(
                    valid_count / float(prepared.depth_geometry.size), 6
                ),
            },
            "configuration": {
                "path": str(self.config_path),
                "axis_direction_enabled": bool(
                    (self.raw_config.get("axis_direction") or {}).get("enabled", False)
                ),
                "full_gripper_motion_collision_enabled": bool(
                    (self.raw_config.get("full_gripper_motion_collision") or {}).get(
                        "enabled", False
                    )
                ),
                "geometry_optimization": self.raw_config.get("geometry_optimization") or {},
                "m38_branch_a": self.raw_config.get("m38_branch_a") or {},
                "m38_branch_b": self.raw_config.get("m38_branch_b") or {},
                "m38_branch_d": self.raw_config.get("m38_branch_d") or {},
                "m38_branch_c": self.raw_config.get("m38_branch_c") or {},
                "hybrid_grasp": self.raw_config.get("hybrid_grasp") or {},
                "side_ring_template": self.raw_config.get("side_ring_template") or {},
            },
            "timing_ms": {
                key: round(float(value), 3) for key, value in timings.items()
            },
            "scene": scene_clean,
            "candidate": selected,
            "files": files,
            "cache_status": self.cache.status(),
        }
        if output_dir is not None and save_flags["geometry_result"]:
            path = output_dir / "online_geometry_result.json"
            files["geometry_result"] = str(path)
            payload["files"] = files
            write_json(path, payload)
        return OnlineProcessResult(payload=payload, overlay_jpeg=overlay_jpeg)
