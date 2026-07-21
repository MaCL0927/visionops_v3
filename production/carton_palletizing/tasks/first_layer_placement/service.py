#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP + trigger-mode WebSocket app for RGB-D carton palletizing."""
from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_palletizing.config import DEFAULT_CONFIG_PATH, load_config
from production.carton_palletizing.tasks.box_grasp_vision.websocket_server import (
    WebSocketJsonServer,
    WebSocketSession,
)
from production.carton_palletizing.tasks.first_layer_placement.algorithm import FirstLayerPlacementAlgorithm
from production.carton_palletizing.tasks.first_layer_placement.trigger_protocol import (
    FAULT_CAMERA_DISCONNECTED,
    FAULT_NONE,
    FAULT_TYPE_CAMERA_DISCONNECTED,
    FAULT_TYPE_NONE,
    FAULT_TYPE_VISION_INFERENCE_ERROR,
    FAULT_VISION_INFERENCE_ERROR,
    normalize_axis_angle,
    protocol_item,
    sample_depth_mm,
    select_held_box,
    select_top_surface_targets,
)


MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def timestamp_ms() -> int:
    return int(time.time() * 1000)


class UpstreamError(ConnectionError):
    pass


class CameraUnavailableError(UpstreamError):
    pass


class RuntimeClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request_json(self, method: str, path: str, body: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8") if method == "POST" else None
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise UpstreamError("Runtime HTTP {}: {}".format(error.code, detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise UpstreamError("无法连接 Runtime: {}".format(getattr(error, "reason", error))) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise UpstreamError("Runtime 响应超过大小限制")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("Runtime 返回非 JSON 内容") from error
        if not isinstance(payload, dict):
            raise UpstreamError("Runtime JSON 顶层必须是对象")
        return payload

    def infer_once(self) -> Dict[str, Any]:
        payload = self.request_json("POST", "/api/runtime/infer_once", {})
        if payload.get("message_type") != "inference_result" or payload.get("status") != "ok":
            raise UpstreamError("Runtime infer_once 未返回成功的 inference_result")
        return payload

    def status(self) -> Dict[str, Any]:
        return self.request_json("GET", "/api/runtime/status")


def decode_depth_png(depth_bytes: bytes) -> "np.ndarray":
    if not depth_bytes:
        raise ValueError("深度图为空")
    encoded = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.size == 0:
        raise ValueError("深度 PNG 解码失败")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError("深度图维度非法: {}".format(depth.shape))
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    return depth


class BridgeDepthClient:
    def __init__(
        self,
        base_url: str,
        health_path: str,
        depth_path: str,
        deproject_path: str,
        timeout_s: float,
        max_depth_age_ms: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.health_path = health_path if health_path.startswith("/") else "/" + health_path
        self.depth_path = depth_path if depth_path.startswith("/") else "/" + depth_path
        self.deproject_path = deproject_path if deproject_path.startswith("/") else "/" + deproject_path
        self.timeout_s = timeout_s
        self.max_depth_age_ms = max(0, int(max_depth_age_ms))

    def _read(self, path: str, max_bytes: int) -> bytes:
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            method="GET",
            headers={"Accept": "application/json,image/png,*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(max_bytes + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise CameraUnavailableError("Camera Bridge HTTP {}: {}".format(error.code, detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CameraUnavailableError("无法连接 Camera Bridge: {}".format(getattr(error, "reason", error))) from error
        if len(raw) > max_bytes:
            raise CameraUnavailableError("Camera Bridge 响应超过大小限制")
        return raw

    def _post_json(self, path: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            "{}{}".format(self.base_url, path),
            data=data,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise CameraUnavailableError("Camera Bridge HTTP {}: {}".format(error.code, detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CameraUnavailableError("无法连接 Camera Bridge: {}".format(getattr(error, "reason", error))) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CameraUnavailableError("Camera Bridge JSON 响应超过大小限制")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CameraUnavailableError("Camera Bridge 返回非 JSON 内容") from error
        if not isinstance(payload, dict):
            raise CameraUnavailableError("Camera Bridge JSON 顶层必须是对象")
        return payload

    def health(self) -> Dict[str, Any]:
        raw = self._read(self.health_path, 1024 * 1024)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CameraUnavailableError("Camera Bridge health 返回非 JSON 内容") from error
        if not isinstance(payload, dict):
            raise CameraUnavailableError("Camera Bridge health 顶层必须是对象")
        return payload

    def get_depth(self) -> Tuple["np.ndarray", Dict[str, Any]]:
        health = self.health()
        if health.get("camera_connected") is False:
            raise CameraUnavailableError("深度相机未连接: {}".format(health.get("last_error") or "camera disconnected"))
        try:
            age_ms = int(health.get("last_depth_age_ms", -1))
        except (TypeError, ValueError, OverflowError):
            age_ms = -1
        if self.max_depth_age_ms > 0 and (age_ms < 0 or age_ms > self.max_depth_age_ms):
            raise CameraUnavailableError("深度帧过旧: age={}ms, max={}ms".format(age_ms, self.max_depth_age_ms))
        raw = self._read(self.depth_path, 32 * 1024 * 1024)
        depth = decode_depth_png(raw)
        return depth, {
            "available": True,
            "last_depth_age_ms": age_ms,
            "camera_connected": health.get("camera_connected"),
            "camera_state": health.get("camera_state"),
        }

    def deproject(self, points: Sequence[Sequence[float]]) -> List[List[float]]:
        response = self._post_json(self.deproject_path, {"points": [list(item[:3]) for item in points]})
        if response.get("ok") is not True:
            raise CameraUnavailableError("相机反投影失败: {}".format(response.get("error") or "unknown"))
        values = response.get("points")
        if not isinstance(values, list) or len(values) != len(points):
            raise CameraUnavailableError("相机反投影结果数量不匹配")
        output = []  # type: List[List[float]]
        for item in values:
            position = item.get("position_camera") if isinstance(item, Mapping) else None
            if not isinstance(position, list) or len(position) < 3 or item.get("valid") is not True:
                output.append([0.0, 0.0, 0.0])
                continue
            try:
                output.append([float(position[0]), float(position[1]), float(position[2])])
            except (TypeError, ValueError, OverflowError):
                output.append([0.0, 0.0, 0.0])
        return output


class AppState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.started_at = time.monotonic()
        self.lock = threading.RLock()
        self.latest_decision = None  # type: Optional[Dict[str, Any]]
        self.latest_gateway_message = None  # type: Optional[Dict[str, Any]]
        self.last_error = None  # type: Optional[Dict[str, Any]]
        self.last_latency_ms = 0.0
        self.counters = {
            "evaluate_attempts": 0,
            "evaluate_success": 0,
            "evaluate_failure": 0,
            "resets": 0,
            "ws_connections": 0,
            "ws_disconnects": 0,
            "trigger_received": 0,
            "trigger_success": 0,
            "trigger_failure": 0,
            "trigger_dropped": 0,
            "remote_detect_region_ignored": 0,
        }

    def record_attempt(self) -> None:
        with self.lock:
            self.counters["evaluate_attempts"] += 1

    def record_success(self, decision: Dict[str, Any]) -> None:
        with self.lock:
            self.latest_decision = decision
            self.last_error = None
            self.counters["evaluate_success"] += 1

    def record_failure(self, error: Exception) -> None:
        with self.lock:
            self.last_error = {"code": type(error).__name__, "message": str(error), "timestamp_ms": timestamp_ms()}
            self.counters["evaluate_failure"] += 1

    def record_reset(self) -> None:
        with self.lock:
            self.latest_decision = None
            self.latest_gateway_message = None
            self.last_error = None
            self.counters["resets"] += 1

    def record_gateway(self, message: Mapping[str, Any], success: bool, latency_ms: float) -> None:
        with self.lock:
            self.latest_gateway_message = deepcopy(dict(message))
            self.last_latency_ms = float(latency_ms)
            self.counters["trigger_success" if success else "trigger_failure"] += 1

    def snapshot(self, websocket: Optional[WebSocketJsonServer] = None) -> Dict[str, Any]:
        with self.lock:
            communication = self.config["task"].get("communication", {})
            return {
                "schema_version": "1.0",
                "message_type": "app_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "app_id": "stack_placement",
                "app_instance_id": "carton_palletizing-stack",
                "component": self.config["component"],
                "device_id": self.config["device_id"],
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "latest_decision": deepcopy(self.latest_decision),
                "latest_gateway_message": deepcopy(self.latest_gateway_message),
                "register_snapshot": [],
                "counters": dict(self.counters),
                "last_error": deepcopy(self.last_error),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "websocket": {
                    "enabled": bool(communication.get("enabled", True)),
                    "clients": websocket.client_count() if websocket is not None else 0,
                    "listen_host": communication.get("websocket", {}).get("listen_host"),
                    "listen_port": communication.get("websocket", {}).get("listen_port"),
                    "path": communication.get("websocket", {}).get("path"),
                },
            }


@dataclass
class TriggerRequest:
    session: Optional[WebSocketSession]
    # Keep the original JSON scalar so trigger_task_id can echo either the
    # legacy symbolic name or the numeric alias used by the robot.
    task_id: Any


class FirstLayerPlacementService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.algorithm = FirstLayerPlacementAlgorithm(config["task"]["algorithm"])
        timeout_s = float(config["app"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeClient(str(config["runtime"]["url"]), timeout_s)
        camera = config.get("camera_bridge", {})
        self.depth_bridge = BridgeDepthClient(
            str(camera.get("base_url") or "http://127.0.0.1:18182"),
            str(camera.get("health_path") or "/health"),
            str(camera.get("depth_path") or "/stream/depth.png"),
            str(camera.get("deproject_path") or "/api/coordinate/deproject"),
            timeout_s,
            int(camera.get("max_depth_age_ms", 1500)),
        )
        self.state = AppState(config)
        self.evaluate_lock = threading.Lock()
        self.allow_injected = bool(config.get("debug", {}).get("allow_injected_runtime_result", False))
        self.accepted_task_types = {
            str(item).strip().lower()
            for item in config.get("runtime", {}).get("accepted_task_types", [])
            if str(item).strip()
        }

        self.communication = config["task"].get("communication", {})
        self.trigger_tasks = self.communication.get("trigger_tasks", {})
        self.place_task_id = str(self.trigger_tasks.get("place_target") or "pallet_place_target")
        self.held_task_id = str(self.trigger_tasks.get("held_box") or "held_box_pose")
        self.place_task_aliases = self._build_task_aliases(
            self.place_task_id,
            self.trigger_tasks.get("place_target_aliases", [1, "1"]),
        )
        self.held_task_aliases = self._build_task_aliases(
            self.held_task_id,
            self.trigger_tasks.get("held_box_aliases", [2, "2"]),
        )
        self.held_selection = self.communication.get("held_box_selection", {})
        self.surface_selection = self.communication.get("surface_target_selection", {})
        self.point_sampling = self.communication.get("point_sampling", {})
        self.trigger_sampling = self.communication.get("trigger_sampling", {})
        self.video = self.communication.get("video", {})
        websocket_config = self.communication.get("websocket", {})
        self.status_enabled = bool(websocket_config.get("status_enabled", True))
        self.status_on_connect = bool(
            websocket_config.get("status_on_connect", self.status_enabled)
        )
        self.status_interval_s = max(
            0.5, float(websocket_config.get("status_interval_s", 1.0))
        )
        self.websocket = WebSocketJsonServer(
            str(websocket_config.get("listen_host") or "0.0.0.0"),
            int(websocket_config.get("listen_port", 9001)),
            str(websocket_config.get("path") or "/vision"),
            self._on_websocket_json,
            on_connect=self._on_websocket_connect,
            on_disconnect=self._on_websocket_disconnect,
            token=str(websocket_config.get("token") or ""),
            max_clients=int(websocket_config.get("max_clients", 4)),
            max_payload_bytes=int(websocket_config.get("max_payload_bytes", 1048576)),
            read_timeout_s=float(websocket_config.get("read_timeout_s", 30.0)),
        )
        self.trigger_queue = queue.Queue(maxsize=int(websocket_config.get("trigger_queue_size", 32)))
        self.stop_event = threading.Event()
        self.trigger_thread = None  # type: Optional[threading.Thread]
        self.status_thread = None  # type: Optional[threading.Thread]
        self.remote_config = self.communication.get("remote_config", {})
        self.allow_remote_confidence_threshold = bool(
            self.remote_config.get("allow_confidence_threshold", True)
        )
        self.dynamic_confidence_threshold = None  # type: Optional[float]
        # Robot-side detect_region is intentionally never applied. Runtime ROI is
        # owned by the VisionOps Web UI and persisted in runtime.roi_config_path.
        # Keep the last received value only for diagnostics.
        self.last_ignored_detect_region = None  # type: Optional[List[float]]
        self.protocol_frame_lock = threading.Lock()
        self.protocol_frame_id = 0

    @staticmethod
    def _task_id_token(value: Any) -> str:
        """Normalize a trigger task ID only for alias matching.

        The original JSON scalar is preserved separately for the response, so
        an integer ``1`` is returned as integer ``1`` while string ``"1"`` is
        returned as string ``"1"``.
        """

        if value is None or isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            if value.is_integer():
                return str(int(value))
            return str(value).strip()
        return str(value).strip()

    @classmethod
    def _build_task_aliases(cls, canonical: str, configured: Any) -> set:
        aliases = {cls._task_id_token(canonical)}
        if isinstance(configured, (list, tuple, set)):
            values = configured
        elif configured is None:
            values = []
        else:
            values = [configured]
        for value in values:
            token = cls._task_id_token(value)
            if token:
                aliases.add(token)
        return aliases

    def _resolve_trigger_task(self, requested: Any) -> Tuple[str, Any]:
        """Resolve legacy names and numeric aliases to an internal task name."""

        token = self._task_id_token(requested)
        if not token:
            raise ValueError("trigger.task_id 不能为空")

        # Preserve the incoming scalar for trigger_task_id correlation.
        if isinstance(requested, bool):
            response_id = token
        elif isinstance(requested, int):
            response_id = int(requested)
        elif isinstance(requested, float) and np.isfinite(requested) and requested.is_integer():
            response_id = int(requested)
        else:
            response_id = token

        if token in self.place_task_aliases:
            return self.place_task_id, response_id
        if token in self.held_task_aliases:
            return self.held_task_id, response_id

        allowed = sorted(self.place_task_aliases | self.held_task_aliases)
        raise ValueError(
            "未知 trigger.task_id={!r}，允许值={}".format(requested, allowed)
        )

    def _validate_runtime_result(self, runtime_result: Mapping[str, Any]) -> None:
        task_type = str(runtime_result.get("task_type") or "").strip().lower()
        if self.accepted_task_types and task_type not in self.accepted_task_types:
            raise ValueError(
                "纸箱摆放 Runtime 必须加载 OBB 模型，当前 task_type={!r}，允许值={}".format(
                    task_type, sorted(self.accepted_task_types)
                )
            )

    def start(self) -> None:
        if not bool(self.communication.get("enabled", True)):
            return
        self.stop_event.clear()
        self.websocket.start()
        self.trigger_thread = threading.Thread(target=self._trigger_loop, name="palletizing-trigger", daemon=True)
        self.trigger_thread.start()
        if self.status_enabled:
            self.status_thread = threading.Thread(
                target=self._status_loop, name="palletizing-status", daemon=True
            )
            self.status_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.websocket.stop()
        for thread in (self.trigger_thread, self.status_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)

    def reset(self) -> Dict[str, Any]:
        with self.evaluate_lock:
            self.algorithm.reset()
            self.state.record_reset()
        return {
            "schema_version": "1.0",
            "message_type": "app_command_result",
            "status": "ok",
            "command": "reset_stack",
            "timestamp_ms": timestamp_ms(),
        }

    def _runtime_result(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        injected = request_body.get("runtime_result")
        if isinstance(injected, Mapping):
            if not self.allow_injected:
                raise ValueError("当前配置不允许注入 runtime_result")
            runtime_result = deepcopy(dict(injected))
        else:
            runtime_result = self.runtime.infer_once()
        self._validate_runtime_result(runtime_result)
        return runtime_result

    def evaluate(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        with self.evaluate_lock:
            self.state.record_attempt()
            try:
                if bool(request_body.get("reset")):
                    self.algorithm.reset()
                    self.state.record_reset()
                runtime_result = self._runtime_result(request_body)
                depth_image = None
                depth_status = {"available": False, "reason": "NOT_REQUIRED"}
                if self.algorithm.needs_depth() or bool(request_body.get("force_depth")):
                    try:
                        depth_image, depth_status = self.depth_bridge.get_depth()
                    except Exception as depth_error:  # depth loss remains visible in decision
                        depth_status = {
                            "available": False,
                            "reason": type(depth_error).__name__,
                            "message": str(depth_error),
                        }
                placement = self.algorithm.evaluate(runtime_result, depth_image, depth_status)
                visualization_result = deepcopy(runtime_result)
                visualization_result["placement"] = placement
                decision = {
                    "schema_version": "1.0",
                    "message_type": "app_decision",
                    "status": "ok",
                    "app_id": "stack_placement",
                    "task": "multi_layer_placement",
                    "device_id": self.config["device_id"],
                    "component": self.config["component"],
                    "timestamp_ms": timestamp_ms(),
                    "frame_id": runtime_result.get("frame_id"),
                    "result_id": runtime_result.get("result_id"),
                    "placement": placement,
                    "visualization_result": visualization_result,
                }
                self.state.record_success(decision)
                return decision
            except Exception as error:
                self.state.record_failure(error)
                raise

    def _frame_timestamp(self, runtime_result: Mapping[str, Any]) -> float:
        try:
            capture_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
        except (TypeError, ValueError, OverflowError):
            capture_ms = 0
        return capture_ms / 1000.0 if capture_ms > 0 else time.time()

    def _image_size(self, runtime_result: Mapping[str, Any]) -> Tuple[int, int]:
        image = runtime_result.get("image") if isinstance(runtime_result.get("image"), Mapping) else {}
        try:
            width = max(1, int(image.get("width") or 1))
            height = max(1, int(image.get("height") or 1))
        except (TypeError, ValueError, OverflowError):
            width, height = 1, 1
        return width, height

    def _sample_and_deproject(
        self,
        depth_image: "np.ndarray",
        center_px: Sequence[float],
        image_width: int,
        image_height: int,
    ) -> Tuple[Dict[str, Any], List[float]]:
        settings = self.point_sampling
        sampled = sample_depth_mm(
            depth_image,
            center_px,
            image_width,
            image_height,
            int(settings.get("roi_radius_px", 6)),
            float(settings.get("percentile", 50.0)),
            int(settings.get("min_valid_pixels", 3)),
            int(settings.get("min_depth_mm", 100)),
            int(settings.get("max_depth_mm", 5000)),
        )
        if not sampled.get("valid"):
            return sampled, [0.0, 0.0, 0.0]
        positions = self.depth_bridge.deproject(
            [[float(center_px[0]), float(center_px[1]), float(sampled["depth_mm"])]]
        )
        return sampled, positions[0] if positions else [0.0, 0.0, 0.0]

    def _base_robot_message(
        self,
        runtime_result: Mapping[str, Any],
        task_id: Any,
        items: Sequence[Mapping[str, Any]],
        fault_code: int = FAULT_NONE,
        fault_type: str = FAULT_TYPE_NONE,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            frame_id = int(runtime_result.get("frame_id") or 0)
        except (TypeError, ValueError, OverflowError):
            frame_id = 0
        with self.protocol_frame_lock:
            if frame_id <= 0:
                self.protocol_frame_id += 1
                frame_id = self.protocol_frame_id
            else:
                self.protocol_frame_id = max(self.protocol_frame_id, frame_id)
        message = {
            "type": "detection",
            "frame_id": frame_id,
            "timestamp": self._frame_timestamp(runtime_result),
            "trigger_task_id": task_id,
            "items": [deepcopy(dict(item)) for item in items],
            "fault_code": int(fault_code),
            "fault_type": str(fault_type),
        }
        if extra:
            message.update(deepcopy(dict(extra)))
        return message

    def _place_target_trigger(
        self,
        runtime_result: Mapping[str, Any],
        depth_image: "np.ndarray",
        depth_status: Mapping[str, Any],
        task_id: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return detected support objects, not a planned placement slot.

        M29 deliberately removes layer/slot planning from the robot-facing
        placement trigger.  When cartons are visible, only the nearest depth
        cluster (the current top layer) is returned.  When no carton is
        visible, the best tray detection is returned.
        """

        snapshot = self.algorithm.detection_candidates(
            runtime_result,
            update_tray_reference=True,
        )
        width, height = self._image_size(runtime_result)
        boxes = list(snapshot.get("boxes", []))
        trays = list(snapshot.get("trays", []))

        raw_box_count = len(boxes)
        raw_tray_count = len(trays)

        # Do not apply WebSocket config.detect_region here. The Runtime has
        # already applied the ROI configured from the VisionOps Web UI. Applying
        # a second robot-controlled pixel ROI caused valid 1280x720 detections
        # to disappear when the robot sent a 640x480 region.
        if self.dynamic_confidence_threshold is not None:
            threshold = float(self.dynamic_confidence_threshold)
            boxes = [item for item in boxes if float(item.get("score") or 0.0) >= threshold]
            trays = [item for item in trays if float(item.get("score") or 0.0) >= threshold]

        confidence_filtered_box_count = len(boxes)
        confidence_filtered_tray_count = len(trays)

        selected, target_kind, diagnostics = select_top_surface_targets(
            boxes,
            trays,
            snapshot.get("tray_polygon"),
            depth_image,
            width,
            height,
            self.surface_selection,
        )
        diagnostics.update({
            "roi_control_source": "visionops_web_runtime",
            "robot_detect_region_applied": False,
            "last_ignored_robot_detect_region": deepcopy(self.last_ignored_detect_region),
            "raw_candidate_box_count": raw_box_count,
            "raw_candidate_tray_count": raw_tray_count,
            "confidence_filtered_box_count": confidence_filtered_box_count,
            "confidence_filtered_tray_count": confidence_filtered_tray_count,
            "dynamic_confidence_threshold": self.dynamic_confidence_threshold,
        })

        if selected:
            points = []  # type: List[List[float]]
            valid_selected = []  # type: List[Mapping[str, Any]]
            for item in selected:
                center = item.get("center") or [0.0, 0.0]
                depth = item.get("depth") if isinstance(item.get("depth"), Mapping) else {}
                if not bool(depth.get("valid")):
                    continue
                points.append(
                    [float(center[0]), float(center[1]), float(depth.get("depth_mm") or 0.0)]
                )
                valid_selected.append(item)
            positions = self.depth_bridge.deproject(points) if points else []
        else:
            valid_selected = []
            positions = []

        items = []  # type: List[Dict[str, Any]]
        for source, position in zip(valid_selected, positions):
            if not any(abs(float(value)) > 1e-9 for value in position):
                continue
            center_px = source.get("center") or [0.0, 0.0]
            items.append(
                protocol_item(
                    len(items),
                    int(source.get("class_id") if source.get("class_id") is not None else (1 if target_kind == "tray" else 0)),
                    float(source.get("score") or 0.0),
                    center_px,
                    position,
                    float(source.get("long_axis_angle_deg") or 0.0),
                )
            )

        if selected and not items:
            raise ValueError("检测到{}目标，但中心深度反投影无有效结果".format(target_kind))
        if diagnostics.get("reason") in {"BOX_DEPTH_UNAVAILABLE", "TRAY_DEPTH_UNAVAILABLE"}:
            raise ValueError("顶层目标深度不可用: {}".format(diagnostics.get("reason")))

        robot = self._base_robot_message(
            runtime_result,
            task_id,
            items,
        )
        visualization = deepcopy(runtime_result)
        visualization["surface_target_selection"] = {
            "task_id": task_id,
            "target_kind": target_kind,
            "diagnostics": deepcopy(diagnostics),
            "selected": [deepcopy(dict(item)) for item in valid_selected],
            "depth_status": deepcopy(dict(depth_status)),
        }
        return robot, {
            "visualization_result": visualization,
            "surface_target_selection": {
                "target_kind": target_kind,
                "selected_count": len(items),
                "diagnostics": diagnostics,
            },
        }

    def _place_target_trigger_sequence(
        self, task_id: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Execute exactly one OBB+depth observation for the M29 surface task.

        The previous implementation sampled several frames to advance the
        slot/layer state machine.  M29 no longer owns placement planning, so a
        trigger maps directly to one fresh inference and one detection reply.
        """

        runtime_result = self.runtime.infer_once()
        self._validate_runtime_result(runtime_result)
        depth_image, depth_status = self.depth_bridge.get_depth()
        robot, detail = self._place_target_trigger(
            runtime_result,
            depth_image,
            depth_status,
            task_id,
        )
        return robot, detail, runtime_result

    def _held_box_trigger(
        self,
        runtime_result: Mapping[str, Any],
        depth_image: "np.ndarray",
        task_id: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        snapshot = self.algorithm.detection_candidates(runtime_result, update_tray_reference=False)
        width, height = self._image_size(runtime_result)
        boxes = list(snapshot.get("boxes", []))
        raw_box_count = len(boxes)
        # Robot-side detect_region is ignored for held-box selection as well;
        # the only spatial ROI is the Runtime ROI configured from the Web UI.
        selected, diagnostics = select_held_box(
            boxes,
            snapshot.get("tray_polygon"),
            depth_image,
            width,
            height,
            self.held_selection,
        )
        diagnostics.update({
            "roi_control_source": "visionops_web_runtime",
            "robot_detect_region_applied": False,
            "last_ignored_robot_detect_region": deepcopy(self.last_ignored_detect_region),
            "raw_candidate_box_count": raw_box_count,
            "dynamic_confidence_threshold": self.dynamic_confidence_threshold,
        })
        items = []  # type: List[Dict[str, Any]]
        sampling = None
        if isinstance(selected, Mapping):
            center_px = selected.get("center") or [0.0, 0.0]
            sampling, position = self._sample_and_deproject(depth_image, center_px, width, height)
            if sampling.get("valid") and any(abs(float(value)) > 1e-9 for value in position):
                confidence = float(selected.get("score") or 0.0)
                if self.dynamic_confidence_threshold is None or confidence >= self.dynamic_confidence_threshold:
                    items.append(
                        protocol_item(
                            0,
                            int(selected.get("class_id") or 0),
                            confidence,
                            center_px,
                            position,
                            float(selected.get("long_axis_angle_deg") or 0.0),
                            {"source_detection_id": str(selected.get("id") or "")},
                        )
                    )
        robot = self._base_robot_message(
            runtime_result,
            task_id,
            items,
            extra={
                "result_state": "HELD_BOX_READY" if items else str(diagnostics.get("reason") or "NO_HELD_BOX"),
            },
        )
        visualization = deepcopy(runtime_result)
        visualization["held_box_selection"] = {
            "task_id": task_id,
            "settings_mode": str(self.held_selection.get("mode") or "nearest_depth"),
            "diagnostics": diagnostics,
            "selected": deepcopy(dict(selected)) if isinstance(selected, Mapping) else None,
            "depth_sample": deepcopy(sampling),
        }
        return robot, {"visualization_result": visualization, "held_box_selection": diagnostics}

    def trigger_once(self, task_id: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        response_task_id = task_id
        runtime_result = {}  # type: Dict[str, Any]
        try:
            canonical_task_id, response_task_id = self._resolve_trigger_task(task_id)
            with self.evaluate_lock:
                if canonical_task_id == self.place_task_id:
                    robot, detail, runtime_result = self._place_target_trigger_sequence(response_task_id)
                elif canonical_task_id == self.held_task_id:
                    runtime_result = self.runtime.infer_once()
                    self._validate_runtime_result(runtime_result)
                    depth_image, _depth_status = self.depth_bridge.get_depth()
                    robot, detail = self._held_box_trigger(runtime_result, depth_image, response_task_id)
                else:  # defensive; _resolve_trigger_task already validates
                    raise ValueError("不支持的内部 trigger 任务: {}".format(canonical_task_id))
                decision = {
                    "schema_version": "1.0",
                    "message_type": "app_decision",
                    "status": "ok",
                    "app_id": "stack_placement",
                    "task": "triggered_carton_palletizing",
                    "device_id": self.config["device_id"],
                    "component": self.config["component"],
                    "timestamp_ms": timestamp_ms(),
                    "frame_id": runtime_result.get("frame_id"),
                    "result_id": runtime_result.get("result_id"),
                    "trigger_task_id": response_task_id,
                    "robot_message": robot,
                }
                decision.update(detail)
                self.state.record_success(decision)
                self.state.record_gateway(robot, True, (time.perf_counter() - started) * 1000.0)
                return decision
        except Exception as error:
            if isinstance(error, CameraUnavailableError):
                fault_code, fault_type = FAULT_CAMERA_DISCONNECTED, FAULT_TYPE_CAMERA_DISCONNECTED
            else:
                fault_code, fault_type = FAULT_VISION_INFERENCE_ERROR, FAULT_TYPE_VISION_INFERENCE_ERROR
            robot = self._base_robot_message(
                runtime_result,
                response_task_id,
                [],
                fault_code=fault_code,
                fault_type=fault_type,
                extra={"error": str(error)},
            )
            self.state.record_failure(error)
            self.state.record_gateway(robot, False, (time.perf_counter() - started) * 1000.0)
            return {
                "schema_version": "1.0",
                "message_type": "app_decision",
                "status": "error",
                "app_id": "stack_placement",
                "task": "triggered_carton_palletizing",
                "timestamp_ms": timestamp_ms(),
                "trigger_task_id": response_task_id,
                "robot_message": robot,
                "error": {"code": type(error).__name__, "message": str(error)},
            }

    def _on_websocket_connect(self, session: WebSocketSession) -> None:
        with self.state.lock:
            self.state.counters["ws_connections"] += 1
        if not self.status_on_connect:
            return
        try:
            session.send_json(self._status_message())
        except (ConnectionError, OSError):
            session.close(1006, "initial status send failed")

    def _on_websocket_disconnect(self, _session: WebSocketSession) -> None:
        with self.state.lock:
            self.state.counters["ws_disconnects"] += 1

    def _on_websocket_json(self, session: WebSocketSession, document: Dict[str, Any]) -> None:
        message_type = str(document.get("type") or "").strip().lower()
        if message_type == "ping":
            session.send_json({"type": "pong", "timestamp": time.time()})
            return
        if message_type == "config":
            threshold = document.get("confidence_threshold")
            if threshold is not None and self.allow_remote_confidence_threshold:
                value = float(threshold)
                if not 0.0 <= value <= 1.0:
                    raise ValueError("confidence_threshold 必须位于0..1")
                self.dynamic_confidence_threshold = value

            # Compatibility only: the robot protocol may still send
            # detect_region, but carton_palletizing never applies it. ROI is
            # controlled exclusively by the VisionOps Web UI / Runtime ROI file.
            region = document.get("detect_region")
            if isinstance(region, list) and len(region) >= 4:
                try:
                    self.last_ignored_detect_region = [float(value) for value in region[:4]]
                except (TypeError, ValueError, OverflowError):
                    self.last_ignored_detect_region = None
                with self.state.lock:
                    self.state.counters["remote_detect_region_ignored"] += 1
            return
        if message_type != "trigger":
            # Unknown application messages are ignored so clients do not reconnect.
            return
        task_id = document.get("task_id")
        if not self._task_id_token(task_id):
            return
        with self.state.lock:
            self.state.counters["trigger_received"] += 1
        try:
            self.trigger_queue.put_nowait(TriggerRequest(session=session, task_id=task_id))
        except queue.Full:
            with self.state.lock:
                self.state.counters["trigger_dropped"] += 1
            message = self._base_robot_message(
                {}, task_id, [], FAULT_VISION_INFERENCE_ERROR, FAULT_TYPE_VISION_INFERENCE_ERROR,
                {"error": "trigger queue full"},
            )
            session.send_json(message)

    def _trigger_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.trigger_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            decision = self.trigger_once(request.task_id)
            message = decision.get("robot_message") if isinstance(decision.get("robot_message"), Mapping) else None
            if request.session is not None and message is not None:
                try:
                    request.session.send_json(message)
                except (ConnectionError, OSError):
                    pass
            self.trigger_queue.task_done()

    def _status_message(self) -> Dict[str, Any]:
        snapshot = self.state.snapshot(self.websocket)
        latest = snapshot.get("latest_decision") if isinstance(snapshot.get("latest_decision"), Mapping) else {}
        model = ""
        try:
            model = str((self.runtime.status().get("model") or ""))
        except Exception:
            model = ""
        return {
            "type": "status",
            "online": True,
            "fps": 0.0,
            "model": model,
            "camera_connected": snapshot.get("health") == "ok",
            "latency_ms": snapshot.get("last_latency_ms", 0.0),
            "error": snapshot.get("last_error", {}).get("message") if isinstance(snapshot.get("last_error"), Mapping) else None,
            "clients": self.websocket.client_count(),
            "last_trigger_task_id": latest.get("trigger_task_id") if isinstance(latest, Mapping) else None,
            "video_url": self.video.get("public_url"),
        }

    def _status_loop(self) -> None:
        if not self.status_enabled:
            return
        while not self.stop_event.wait(self.status_interval_s):
            if self.websocket.client_count() <= 0:
                continue
            self.websocket.broadcast_json(self._status_message())


class AppHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], service: FirstLayerPlacementService) -> None:
        self.app_service = service
        super().__init__(address, AppRequestHandler)


class AppRequestHandler(BaseHTTPRequestHandler):
    server: AppHttpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print("[{}] {} {}".format(self.log_date_time_string(), self.address_string(), fmt % args))

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str, detail: Any = None) -> None:
        self._json(status, {
            "schema_version": "1.0",
            "message_type": "app_error",
            "status": "error",
            "timestamp_ms": timestamp_ms(),
            "error": {"code": code, "message": message, "detail": detail, "recoverable": True},
        })

    def _body(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法")
            return None
        if not 0 <= length <= MAX_REQUEST_BYTES:
            self._error(413, "REQUEST_TOO_LARGE", "请求体超过限制")
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "INVALID_JSON", "请求体必须是 JSON 对象")
            return None
        if not isinstance(payload, dict):
            self._error(400, "INVALID_JSON", "请求体顶层必须是对象")
            return None
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        service = self.server.app_service
        if path == "/health":
            status = service.state.snapshot(service.websocket)
            self._json(200, {
                "schema_version": "1.0",
                "message_type": "app_health",
                "status": "ok",
                "health": status["health"],
                "app_id": status["app_id"],
                "device_id": status["device_id"],
                "timestamp_ms": timestamp_ms(),
            })
        elif path == "/api/app/status":
            self._json(200, service.state.snapshot(service.websocket))
        elif path == "/api/app/registers":
            self._json(200, {"schema_version": "1.0", "message_type": "app_register_snapshot", "status": "ok", "registers": []})
        elif path == "/api/app/latest_decision":
            latest = service.state.snapshot(service.websocket)["latest_decision"]
            if latest is None:
                self._error(404, "LATEST_DECISION_NOT_FOUND", "尚未生成纸箱堆垛决策")
            else:
                self._json(200, latest)
        elif path == "/api/app/latest_gateway_message":
            latest = service.state.snapshot(service.websocket)["latest_gateway_message"]
            if latest is None:
                self._error(404, "GATEWAY_MESSAGE_NOT_FOUND", "尚未生成机器人通信消息")
            else:
                self._json(200, latest)
        elif path == "/api/gateway/status":
            self._json(200, service.state.snapshot(service.websocket)["websocket"])
        elif path == "/api/gateway/registers":
            self._json(200, {"schema_version": "1.0", "message_type": "gateway_register_snapshot", "status": "ok", "registers": []})
        else:
            self._error(404, "ROUTE_NOT_FOUND", "接口不存在")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        body = self._body()
        if body is None:
            return
        service = self.server.app_service
        try:
            if path == "/api/app/evaluate_once":
                self._json(200, service.evaluate(body))
            elif path == "/api/app/trigger":
                self._json(200, service.trigger_once(body.get("task_id")))
            elif path == "/api/app/reset":
                self._json(200, service.reset())
            else:
                self._error(404, "ROUTE_NOT_FOUND", "接口不存在")
        except ValueError as error:
            self._error(400, "INVALID_REQUEST", str(error))
        except UpstreamError as error:
            self._error(502, "RUNTIME_UNAVAILABLE", "纸箱摆放应用无法取得上游结果", str(error))
        except Exception as error:  # noqa: BLE001
            self._error(500, "EVALUATION_FAILED", "多层堆垛计算失败", str(error))


def run(config: Mapping[str, Any]) -> int:
    service = FirstLayerPlacementService(config)
    server = AppHttpServer((str(config["app"]["listen_host"]), int(config["app"]["listen_port"])), service)
    stopping = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    service.start()
    websocket = config["task"].get("communication", {}).get("websocket", {})
    print(
        "Carton palletizing app listening on {}:{}, Runtime={}, WebSocket={}:{}{}".format(
            config["app"]["listen_host"],
            config["app"]["listen_port"],
            config["runtime"]["url"],
            websocket.get("listen_host", "0.0.0.0"),
            websocket.get("listen_port", 9001),
            websocket.get("path", "/vision"),
        )
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        service.stop()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="纸箱托盘多层 RGB-D 摆放与触发通信应用")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
