#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP + WebSocket service for the detergent OBB grasp task.

The WebSocket transport deliberately follows the existing VisionOps external-box
contract.  It is WebSocket JSON over TCP, not a new raw-TCP framing protocol.
Every completed inference result is pushed; there is no independent push-Hz
setting that can throttle the actual producer FPS.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.runtime_client import HttpClient
from production.common.runtime_ipc import RuntimeIpcClient
from production.detergent_grasp.config import DEFAULT_CONFIG_PATH, load_config
from production.detergent_grasp.tasks.detergent_grasp_vision.algorithm import (
    DetergentGraspAlgorithm,
    DetergentGraspResult,
)
from production.detergent_grasp.tasks.detergent_grasp_vision.websocket_server import (
    WebSocketJsonServer,
    WebSocketSession,
)

MAX_HTTP_BODY = 1024 * 1024
FAULT_NONE = 0
FAULT_CAMERA_DISCONNECTED = 3101
FAULT_VISION_INFERENCE_ERROR = 3201
FAULT_TYPE_NONE = "NONE"
FAULT_TYPE_CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
FAULT_TYPE_VISION_INFERENCE_ERROR = "VISION_INFERENCE_ERROR"
MIN_PRODUCTION_INFERENCE_FPS = 0.1
MAX_PRODUCTION_INFERENCE_FPS = 30.0
INFERENCE_SETTINGS_SCHEMA_VERSION = "2.0"


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _path_models(result: Mapping[str, Any]) -> set[str]:
    model = result.get("model") if isinstance(result.get("model"), Mapping) else {}
    values: set[str] = set()
    for key in ("model_id", "model_name", "package_id", "model_dir", "path"):
        value = model.get(key)
        if value:
            values.add(str(value))
            values.add(Path(str(value)).name)
    return values


@dataclass(frozen=True)
class TriggerRequest:
    session: WebSocketSession
    request_id: Optional[object]
    trigger_task_id: Optional[object]


@dataclass
class InferencePacket:
    """One Runtime result moving from the inference stage to postprocess.

    Continuous packets are latest-only and may be replaced before postprocess.
    Explicit robot triggers carry ``trigger`` and are never intentionally dropped.
    """

    frame_id: int
    request_id: Optional[object]
    trigger_task_id: Optional[object]
    started_timestamp: float
    started_monotonic: float
    runtime_raw: bytes = b""
    runtime_result: Optional[Dict[str, Any]] = None
    runtime_lock_wait_ms: float = 0.0
    runtime_request_ms: float = 0.0
    runtime_connect_ms: float = 0.0
    runtime_send_ms: float = 0.0
    runtime_headers_wait_ms: float = 0.0
    runtime_body_read_ms: float = 0.0
    runtime_transport: str = "unknown"
    runtime_server_queue_ms: float = 0.0
    runtime_server_route_ms: float = 0.0
    runtime_json_decode_ms: float = 0.0
    runtime_internal_ms: float = 0.0
    inference_stage_ms: float = 0.0
    runtime_response_bytes: int = 0
    inference_completed_monotonic: float = 0.0
    error: Optional[Exception] = None
    trigger: Optional[TriggerRequest] = None
    continuous: bool = False


class CameraUnavailableError(RuntimeError):
    pass


class ServiceState:
    """Thread-safe live state and rolling App timing statistics."""

    def __init__(self, config: Mapping[str, Any], configured_fps: float) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.continuous_enabled = bool(config["websocket"].get("auto_start", True))
        self.busy = False
        self.inference_busy = False
        self.postprocess_busy = False
        self.frame_id = 0
        self.configured_fps = float(configured_fps)
        self.latest_decision: Optional[Dict[str, Any]] = None
        self.latest_robot_message: Optional[Dict[str, Any]] = None
        self.latest_runtime_result: Optional[Dict[str, Any]] = None
        self.last_error: Optional[Dict[str, Any]] = None
        self.last_latency_ms = 0.0
        self.last_app_timing: Dict[str, Any] = {}
        self.completion_times: deque[float] = deque(maxlen=120)
        self.inference_stage_times: deque[float] = deque(maxlen=120)
        self.latency_samples: deque[float] = deque(maxlen=120)
        self.timing_samples: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=120))
        self.counters: Dict[str, int] = defaultdict(int)

    def next_frame_id(self) -> int:
        with self.lock:
            self.frame_id += 1
            return self.frame_id

    def set_continuous(self, enabled: bool) -> None:
        with self.lock:
            self.continuous_enabled = bool(enabled)

    def set_configured_fps(self, fps: float) -> None:
        with self.lock:
            self.configured_fps = float(fps)

    def begin_inference(self) -> None:
        with self.lock:
            self.inference_busy = True
            self.busy = True
            self.counters["inference_requests"] += 1

    def end_inference(self, completed: bool = True) -> None:
        with self.lock:
            self.inference_busy = False
            self.busy = self.postprocess_busy
            if completed:
                self.inference_stage_times.append(time.monotonic())
                self.counters["inference_stage_completed"] += 1

    def begin_postprocess(self) -> None:
        with self.lock:
            self.postprocess_busy = True
            self.busy = True
            self.counters["postprocess_requests"] += 1

    def end_postprocess(self) -> None:
        with self.lock:
            self.postprocess_busy = False
            self.busy = self.inference_busy

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * percentile) - 1)))
        return round(ordered[index], 3)

    @staticmethod
    def _rate(times: Sequence[float]) -> float:
        if len(times) < 2:
            return 0.0
        elapsed = float(times[-1]) - float(times[0])
        return round((len(times) - 1) / elapsed, 3) if elapsed > 0.0 else 0.0

    def fps(self) -> float:
        with self.lock:
            return self._rate(list(self.completion_times))

    def inference_stage_fps(self) -> float:
        with self.lock:
            return self._rate(list(self.inference_stage_times))

    def _record_timing_locked(self, timing: Mapping[str, Any]) -> None:
        self.last_app_timing = dict(timing)
        for key, value in timing.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.timing_samples[str(key)].append(float(value))

    def success(
        self,
        decision: Mapping[str, Any],
        robot_message: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        started_monotonic: float,
        app_timing: Dict[str, Any],
    ) -> float:
        store_started = time.perf_counter()
        with self.lock:
            latency_ms = round((time.monotonic() - started_monotonic) * 1000.0, 3)
            if isinstance(robot_message, dict):
                robot_message["latency_ms"] = latency_ms
            # Results are immutable after publication.  Keep shallow top-level
            # copies and deepcopy only when an HTTP snapshot is requested.
            self.latest_decision = dict(decision)
            self.latest_robot_message = dict(robot_message)
            self.latest_runtime_result = dict(runtime_result)
            self.last_error = None
            app_timing["state_store_ms"] = round((time.perf_counter() - store_started) * 1000.0, 3)
            app_timing["pipeline_age_ms"] = latency_ms
            app_timing["total_ms"] = latency_ms
            self.last_latency_ms = latency_ms
            self.completion_times.append(time.monotonic())
            self.latency_samples.append(latency_ms)
            self._record_timing_locked(app_timing)
            self.counters["inference_success"] += 1
            self.busy = self.inference_busy or self.postprocess_busy
            return latency_ms

    def failure(
        self,
        decision: Mapping[str, Any],
        robot_message: Mapping[str, Any],
        error: Exception,
        started_monotonic: float,
        app_timing: Dict[str, Any],
    ) -> float:
        store_started = time.perf_counter()
        with self.lock:
            latency_ms = round((time.monotonic() - started_monotonic) * 1000.0, 3)
            if isinstance(robot_message, dict):
                robot_message["latency_ms"] = latency_ms
            self.latest_decision = dict(decision)
            self.latest_robot_message = dict(robot_message)
            app_timing["state_store_ms"] = round((time.perf_counter() - store_started) * 1000.0, 3)
            app_timing["pipeline_age_ms"] = latency_ms
            app_timing["total_ms"] = latency_ms
            self.last_latency_ms = latency_ms
            self.latency_samples.append(latency_ms)
            self._record_timing_locked(app_timing)
            self.last_error = {
                "code": type(error).__name__,
                "message": str(error),
                "timestamp_ms": _timestamp_ms(),
            }
            self.counters["inference_failure"] += 1
            self.busy = self.inference_busy or self.postprocess_busy
            return latency_ms

    def snapshot(self, websocket: WebSocketJsonServer) -> Dict[str, Any]:
        with self.lock:
            latency_values = list(self.latency_samples)
            timing_stats = {
                key: {
                    "avg": round(sum(values) / len(values), 3),
                    "p50": self._percentile(list(values), 0.50),
                    "p95": self._percentile(list(values), 0.95),
                    "max": round(max(values), 3),
                    "samples": len(values),
                }
                for key, values in self.timing_samples.items()
                if values
            }
            return {
                "schema_version": "1.0",
                "message_type": "detergent_grasp_service_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "timestamp_ms": _timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "inference_busy": self.inference_busy,
                "postprocess_busy": self.postprocess_busy,
                "continuous_enabled": self.continuous_enabled,
                "detection_fps": self._rate(list(self.completion_times)),
                "inference_stage_fps": self._rate(list(self.inference_stage_times)),
                "configured_detection_fps": round(self.configured_fps, 6),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "latency_ms": {
                    "latest": round(self.last_latency_ms, 3),
                    "avg": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
                    "p50": self._percentile(latency_values, 0.50),
                    "p95": self._percentile(latency_values, 0.95),
                    "max": round(max(latency_values), 3) if latency_values else 0.0,
                    "samples": len(latency_values),
                },
                "last_app_timing": deepcopy(self.last_app_timing),
                "app_timing_stats": timing_stats,
                "websocket": {
                    "listen_host": self.config["websocket"]["listen_host"],
                    "listen_port": self.config["websocket"]["listen_port"],
                    "path": self.config["websocket"]["path"],
                    "clients": websocket.client_count(),
                },
                "video": {
                    "type": "mjpeg",
                    "url": self.config["video"]["public_url"],
                    "sync": "soft",
                },
                "runtime_url": self.config["runtime"]["url"],
                "latest_decision": deepcopy(self.latest_decision),
                "latest_gateway_message": deepcopy(self.latest_robot_message),
                "last_error": deepcopy(self.last_error),
                "counters": dict(self.counters),
            }


class DetergentGraspVisionService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        timeout_s = float(config["app"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeIpcClient(
            str(config["runtime"]["url"]),
            timeout_s,
            config.get("runtime_ipc") if isinstance(config.get("runtime_ipc"), Mapping) else {},
        )
        self.http = HttpClient(timeout_s=timeout_s)
        self.algorithm = DetergentGraspAlgorithm(config["algorithm"])
        self.inference_settings_path = Path(str(config["app"]["inference_settings_path"]))
        self.production_fps_lock = threading.Lock()
        self.production_inference_fps = float(config["app"].get("default_production_inference_fps", 15.0))
        self.production_fps_source = "default"
        self._load_production_fps_override()
        self.state = ServiceState(config, self.production_inference_fps)
        # Manual HTTP requests serialize with one another.  Runtime access has a
        # separate lock so one RKNN context remains safe while inference and
        # CPU postprocess overlap across different frames.
        self.execution_lock = threading.Lock()
        self.runtime_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), Mapping) else {}
        self.pipeline_enabled = bool(pipeline.get("enabled", True))
        self.pipeline_max_result_age_ms = max(1, int(pipeline.get("max_result_age_ms", 500)))
        self.result_queue: "queue.Queue[InferencePacket]" = queue.Queue(
            maxsize=max(1, int(pipeline.get("result_queue_size", 1)))
        )
        self.worker_thread: Optional[threading.Thread] = None
        self.postprocess_thread: Optional[threading.Thread] = None
        self.status_thread: Optional[threading.Thread] = None
        self.manual_request_id = 0
        self.trigger_queue: "queue.Queue[TriggerRequest]" = queue.Queue(
            maxsize=int(config["websocket"].get("trigger_queue_size", 32))
        )
        debug = config.get("debug") if isinstance(config.get("debug"), Mapping) else {}
        self.debug_enabled = bool(debug.get("save_every_trigger", False))
        self.debug_root = Path(str(debug.get("save_root", "/tmp/visionops_v3/detergent_grasp/latest")))
        self.debug_lock = threading.Lock()
        ws = config["websocket"]
        self.websocket = WebSocketJsonServer(
            host=str(ws["listen_host"]),
            port=int(ws["listen_port"]),
            path=str(ws["path"]),
            on_json=self._on_ws_json,
            on_connect=self._on_ws_connect,
            on_disconnect=self._on_ws_disconnect,
            token=str(ws.get("token") or ""),
            max_clients=int(ws.get("max_clients", 4)),
            max_payload_bytes=int(ws.get("max_payload_bytes", 1048576)),
            read_timeout_s=float(ws.get("read_timeout_s", 30.0)),
        )

    def _load_production_fps_override(self) -> None:
        try:
            payload = json.loads(self.inference_settings_path.read_text(encoding="utf-8"))
            if str(payload.get("schema_version") or "") != INFERENCE_SETTINGS_SCHEMA_VERSION:
                return
            fps = float(payload.get("production_inference_fps"))
            if MIN_PRODUCTION_INFERENCE_FPS <= fps <= MAX_PRODUCTION_INFERENCE_FPS:
                self.production_inference_fps = fps
                self.production_fps_source = "persistent"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    def _persist_production_fps(self, fps: float) -> None:
        self.inference_settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.inference_settings_path.with_suffix(self.inference_settings_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": INFERENCE_SETTINGS_SCHEMA_VERSION,
                    "production_inference_fps": fps,
                    "detection_fps": fps,
                    "source": "app_inference_settings_api",
                    "updated_at_ms": _timestamp_ms(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(self.inference_settings_path))

    def production_fps(self) -> float:
        with self.production_fps_lock:
            return float(self.production_inference_fps)

    def set_production_fps(self, value: object) -> Dict[str, Any]:
        try:
            fps = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("detection_fps 必须是数字") from error
        if not MIN_PRODUCTION_INFERENCE_FPS <= fps <= MAX_PRODUCTION_INFERENCE_FPS:
            raise ValueError("detection_fps 必须位于 0.1..30")
        with self.production_fps_lock:
            self.production_inference_fps = fps
            self.production_fps_source = "persistent"
            self._persist_production_fps(fps)
        self.state.set_configured_fps(fps)
        self.wakeup.set()
        return self.inference_settings()

    def inference_settings(self) -> Dict[str, Any]:
        configured = self.production_fps()
        return {
            "schema_version": INFERENCE_SETTINGS_SCHEMA_VERSION,
            "message_type": "app_inference_settings",
            "status": "ok",
            "app_id": "detergent_grasp_vision",
            "production_inference_fps": configured,
            "detection_fps": configured,
            "actual_inference_fps": self.state.fps(),
            "settings_source": self.production_fps_source,
            "push_mode": "every_completed_result",
            "continuous_enabled": self.state.continuous_enabled,
            "timestamp_ms": _timestamp_ms(),
        }

    def producer_metadata(self) -> Dict[str, Any]:
        return {
            "configured_fps": self.production_fps(),
            "actual_fps": self.state.fps(),
            "push_mode": "every_completed_result",
        }

    def pipeline_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.pipeline_enabled,
            "architecture": "inference_producer_latest_only_postprocess",
            "result_queue_size": self.result_queue.qsize(),
            "result_queue_capacity": self.result_queue.maxsize,
            "max_result_age_ms": self.pipeline_max_result_age_ms,
            "inference_thread_alive": bool(self.worker_thread and self.worker_thread.is_alive()),
            "postprocess_thread_alive": bool(self.postprocess_thread and self.postprocess_thread.is_alive()),
            "runtime_ipc": self.runtime.transport_status(),
        }

    def start(self) -> None:
        self.websocket.start()
        self.worker_thread = threading.Thread(target=self._inference_loop, name="detergent-grasp-inference", daemon=True)
        self.postprocess_thread = threading.Thread(
            target=self._postprocess_loop, name="detergent-grasp-postprocess", daemon=True
        )
        self.status_thread = threading.Thread(target=self._status_loop, name="detergent-grasp-status", daemon=True)
        self.worker_thread.start()
        self.postprocess_thread.start()
        self.status_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wakeup.set()
        self.websocket.stop()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        if self.postprocess_thread is not None:
            self.postprocess_thread.join(timeout=5.0)
        if self.status_thread is not None:
            self.status_thread.join(timeout=3.0)

    @staticmethod
    def _valid_request_id(value: object) -> bool:
        return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) != ""

    def _ack(
        self,
        session: WebSocketSession,
        request_type: str,
        success: bool,
        request_id: Optional[object] = None,
        **extra: Any,
    ) -> None:
        document: Dict[str, Any] = {
            "type": "ack",
            "request_type": request_type,
            "success": bool(success),
            "timestamp": time.time(),
        }
        if request_id is not None:
            document["request_id"] = request_id
        document.update(extra)
        session.send_json(document)

    def _on_ws_connect(self, session: WebSocketSession) -> None:
        self.state.counters["connections"] += 1
        try:
            session.send_json(self._status_message())
        except OSError:
            session.close(1006, "initial status send failed")
        self.wakeup.set()

    def _on_ws_disconnect(self, _session: WebSocketSession) -> None:
        self.state.counters["disconnects"] += 1

    def _queue_trigger(
        self,
        session: WebSocketSession,
        request_id: Optional[object],
        trigger_task_id: Optional[object],
        request_type: str,
    ) -> None:
        try:
            self.trigger_queue.put_nowait(
                TriggerRequest(session=session, request_id=request_id, trigger_task_id=trigger_task_id)
            )
        except queue.Full:
            self._ack(session, request_type, False, request_id, error="trigger queue full")
            return
        self._ack(
            session,
            request_type,
            True,
            request_id,
            queued=True,
            trigger_task_id=trigger_task_id,
        )
        self.wakeup.set()

    def _trigger_task_allowed(self, value: object) -> bool:
        configured = self.config["websocket"].get("trigger_task_ids", [])
        return any(str(value) == str(item) for item in configured)

    def _on_ws_json(self, session: WebSocketSession, document: Dict[str, Any]) -> None:
        message_type = str(document.get("type") or "").strip().lower()
        if message_type == "control":
            command = str(document.get("command") or "").strip().lower()
            request_id = document.get("request_id")
            if command == "start":
                self.state.set_continuous(True)
                self._ack(session, "control", True, request_id, command=command)
                self.wakeup.set()
                return
            if command == "stop":
                self.state.set_continuous(False)
                self._ack(session, "control", True, request_id, command=command)
                self.wakeup.set()
                return
            if command == "trigger":
                if not self._valid_request_id(request_id):
                    self._ack(session, "control", False, request_id, command=command, error="trigger 必须携带 request_id")
                    return
                self._queue_trigger(session, request_id, None, "control")
                return
            self._ack(session, "control", False, request_id, command=command, error="unsupported command")
            return
        if message_type == "trigger":
            task_id = document.get("task_id")
            request_id = document.get("request_id")
            if not self._trigger_task_allowed(task_id):
                self._ack(session, "trigger", False, request_id, trigger_task_id=task_id, error="unsupported task_id")
                return
            self._queue_trigger(session, request_id, task_id, "trigger")
            return
        if message_type == "ping":
            session.send_json({"type": "pong", "timestamp": time.time()})
            return
        if message_type == "config":
            self._ack(
                session,
                "config",
                False,
                document.get("request_id"),
                error="ROI/threshold 仅由 VisionOps Web 与任务配置管理",
            )
            return
        self._ack(session, message_type or "unknown", False, document.get("request_id"), error="unsupported message type")

    def _validate_runtime(self, result: Mapping[str, Any]) -> None:
        task_type = str(result.get("task_type") or "").strip().lower()
        accepted_types = set(self.config["runtime"].get("accepted_task_types", []))
        if accepted_types and task_type not in accepted_types:
            raise ValueError("Runtime task_type={!r} 不在白名单 {}".format(task_type, sorted(accepted_types)))
        accepted_models = set(self.config["runtime"].get("accepted_model_ids", [])) | set(
            self.config["runtime"].get("accepted_model_names", [])
        )
        if accepted_models and not (accepted_models & _path_models(result)):
            raise ValueError(
                "Runtime 当前模型不在白名单: current={}, accepted={}".format(
                    sorted(_path_models(result)), sorted(accepted_models)
                )
            )

    def _bridge_health(self) -> Dict[str, Any]:
        bridge = self.config["camera_bridge"]
        url = str(bridge["base_url"]).rstrip("/") + str(bridge.get("health_path") or "/health")
        try:
            raw = self.http.request("GET", url).json()
        except Exception as error:
            return {
                "connected": False,
                "camera_state": "offline",
                "last_color_age_ms": -1,
                "error": str(error),
                "raw": {},
            }
        try:
            age = int(raw.get("last_color_age_ms", -1))
        except (TypeError, ValueError, OverflowError):
            age = -1
        stale_ms = max(500, int(bridge.get("stale_ms", 5000)))
        explicit = raw.get("camera_connected")
        if isinstance(explicit, bool):
            connected = explicit and 0 <= age <= stale_ms
        else:
            connected = bool(raw.get("camera_started")) and 0 <= age <= stale_ms
        return {
            "connected": connected,
            "camera_state": str(raw.get("camera_state") or ("running" if connected else "offline")),
            "last_color_age_ms": age,
            "error": "" if connected else str(raw.get("last_error") or raw.get("error") or "camera frame stale"),
            "raw": raw,
        }

    def _require_camera_ready(self) -> Dict[str, Any]:
        status = self._bridge_health()
        if not status["connected"]:
            raise CameraUnavailableError(status["error"] or "camera unavailable")
        return status

    def _fault_for_error(self, error: Exception) -> tuple[int, str]:
        text = str(error).lower()
        if isinstance(error, CameraUnavailableError) or any(
            token in text for token in ("camera", "frame stale", "snapshot", "相机", "图像帧")
        ):
            return FAULT_CAMERA_DISCONNECTED, FAULT_TYPE_CAMERA_DISCONNECTED
        # The hot inference path intentionally does not poll Bridge /health on
        # every frame.  Only after a failure do we query Runtime/Bridge to map
        # the stable external fault code without reducing normal FPS.
        try:
            runtime_status = self.runtime.status()
            if runtime_status.get("camera_connected") is False:
                return FAULT_CAMERA_DISCONNECTED, FAULT_TYPE_CAMERA_DISCONNECTED
        except Exception:
            pass
        try:
            if not self._bridge_health().get("connected"):
                return FAULT_CAMERA_DISCONNECTED, FAULT_TYPE_CAMERA_DISCONNECTED
        except Exception:
            pass
        return FAULT_VISION_INFERENCE_ERROR, FAULT_TYPE_VISION_INFERENCE_ERROR

    def _robot_message(
        self,
        frame_id: int,
        timestamp: float,
        result: Optional[DetergentGraspResult],
        latency_ms: float,
        request_id: Optional[object],
        trigger_task_id: Optional[object],
        fault_code: int = FAULT_NONE,
        fault_type: str = FAULT_TYPE_NONE,
        runtime_result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        image_width = result.image_width if result is not None else int(self.config["algorithm"]["image"]["width"])
        image_height = result.image_height if result is not None else int(self.config["algorithm"]["image"]["height"])
        document: Dict[str, Any] = {
            "type": "detection",
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "task_id": str(self.config.get("task_id") or "detergent_grasp"),
            "items": deepcopy(result.items) if result is not None else [],
            "image": {"width": image_width, "height": image_height},
            "coordinate_frame": "image",
            "coordinate_unit": "pixel",
            "video_url": self.config["video"]["public_url"],
            "video_sync": "soft",
            "latency_ms": round(float(latency_ms), 3),
            "fault_code": int(fault_code),
            "fault_type": str(fault_type),
            "source": {
                "runtime_frame_id": runtime_result.get("frame_id") if runtime_result else None,
                "runtime_result_id": runtime_result.get("result_id") if runtime_result else None,
            },
        }
        if request_id is not None:
            document["request_id"] = request_id
        if trigger_task_id is not None:
            document["trigger_task_id"] = trigger_task_id
        return document

    @staticmethod
    def _runtime_internal_ms(runtime_result: Mapping[str, Any]) -> float:
        timing = runtime_result.get("timing") if isinstance(runtime_result.get("timing"), Mapping) else {}
        try:
            return float(timing.get("total_ms") or 0.0)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def _run_inference_stage(
        self,
        request_id: Optional[object] = None,
        trigger_task_id: Optional[object] = None,
        trigger: Optional[TriggerRequest] = None,
        continuous: bool = False,
    ) -> InferencePacket:
        packet = InferencePacket(
            frame_id=self.state.next_frame_id(),
            request_id=request_id,
            trigger_task_id=trigger_task_id,
            started_timestamp=time.time(),
            started_monotonic=time.monotonic(),
            trigger=trigger,
            continuous=bool(continuous),
        )
        self.state.begin_inference()
        stage_started = time.perf_counter()
        completed = False
        try:
            lock_wait_started = time.perf_counter()
            with self.runtime_lock:
                packet.runtime_lock_wait_ms = round(
                    (time.perf_counter() - lock_wait_started) * 1000.0, 3
                )
                # Keep the producer stage limited to Runtime I/O.  The raw HTTP
                # body is handed to the CPU postprocess thread, which performs
                # UTF-8/JSON decoding while Runtime starts the next frame.
                response = self.runtime.infer_once_raw()
            packet.runtime_raw = response.body
            packet.runtime_request_ms = round(response.total_ms, 3)
            packet.runtime_connect_ms = round(response.connect_ms, 3)
            packet.runtime_send_ms = round(response.send_ms, 3)
            packet.runtime_headers_wait_ms = round(response.headers_wait_ms, 3)
            packet.runtime_body_read_ms = round(response.body_read_ms, 3)
            packet.runtime_transport = str(response.transport)
            packet.runtime_server_queue_ms = round(
                response.header_float("x-visionops-http-queue-ms"), 3
            )
            packet.runtime_server_route_ms = round(
                response.header_float("x-visionops-http-route-ms"), 3
            )
            packet.runtime_response_bytes = len(response.body)
            completed = True
        except Exception as error:
            packet.error = error
            completed = True
        finally:
            packet.inference_stage_ms = round((time.perf_counter() - stage_started) * 1000.0, 3)
            packet.inference_completed_monotonic = time.monotonic()
            self.state.end_inference(completed=completed)
        return packet

    def _dispatch_packet(self, packet: InferencePacket, robot: Mapping[str, Any]) -> tuple[float, int]:
        started = time.perf_counter()
        sent = 0
        if packet.trigger is not None:
            try:
                packet.trigger.session.send_json(robot)
                sent = 1
            except (OSError, ConnectionError):
                self.state.counters["websocket_send_failures"] += 1
        elif packet.continuous and self.websocket.client_count() > 0:
            sent = self.websocket.broadcast_json(robot)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self.state.counters["websocket_messages_sent"] += int(sent)
        return elapsed_ms, sent

    def _complete_packet(self, packet: InferencePacket, dispatch: bool = False) -> Dict[str, Any]:
        self.state.begin_postprocess()
        postprocess_started = time.perf_counter()
        timing: Dict[str, Any] = {
            "runtime_lock_wait_ms": round(packet.runtime_lock_wait_ms, 3),
            "runtime_request_ms": round(packet.runtime_request_ms, 3),
            "runtime_roundtrip_ms": round(packet.runtime_request_ms, 3),
            "runtime_connect_ms": round(packet.runtime_connect_ms, 3),
            "runtime_send_ms": round(packet.runtime_send_ms, 3),
            "runtime_headers_wait_ms": round(packet.runtime_headers_wait_ms, 3),
            "runtime_body_read_ms": round(packet.runtime_body_read_ms, 3),
            "runtime_transport": packet.runtime_transport,
            "runtime_server_queue_ms": round(packet.runtime_server_queue_ms, 3),
            "runtime_server_route_ms": round(packet.runtime_server_route_ms, 3),
            "runtime_json_decode_ms": 0.0,
            "runtime_internal_ms": round(packet.runtime_internal_ms, 3),
            "runtime_outside_ms": 0.0,
            "runtime_non_route_ms": round(
                max(
                    0.0,
                    packet.runtime_request_ms
                    - packet.runtime_server_queue_ms
                    - packet.runtime_server_route_ms,
                ),
                3,
            ),
            "runtime_response_bytes": int(packet.runtime_response_bytes),
            "inference_stage_ms": round(packet.inference_stage_ms, 3),
            "result_queue_wait_ms": round(
                max(0.0, (time.monotonic() - packet.inference_completed_monotonic) * 1000.0), 3
            ) if packet.inference_completed_monotonic > 0.0 else 0.0,
            "runtime_validation_ms": 0.0,
            "algorithm_ms": 0.0,
            "robot_message_build_ms": 0.0,
            "visualization_build_ms": 0.0,
            "decision_build_ms": 0.0,
            "websocket_send_ms": 0.0,
            "websocket_clients_sent": 0,
            "debug_enqueue_ms": 0.0,
            "postprocess_stage_ms": 0.0,
            "state_store_ms": 0.0,
            "pipeline_age_ms": 0.0,
            "total_ms": 0.0,
        }
        try:
            if packet.error is not None:
                raise packet.error
            runtime_result = packet.runtime_result
            if runtime_result is None:
                decode_started = time.perf_counter()
                runtime_result = self.runtime.decode_inference(packet.runtime_raw)
                packet.runtime_json_decode_ms = round(
                    (time.perf_counter() - decode_started) * 1000.0, 3
                )
                packet.runtime_result = runtime_result
            packet.runtime_internal_ms = round(self._runtime_internal_ms(runtime_result), 3)
            timing["runtime_json_decode_ms"] = packet.runtime_json_decode_ms
            timing["runtime_internal_ms"] = packet.runtime_internal_ms
            timing["runtime_outside_ms"] = round(
                max(0.0, packet.runtime_request_ms - packet.runtime_internal_ms), 3
            )

            validation_started = time.perf_counter()
            self._validate_runtime(runtime_result)
            timing["runtime_validation_ms"] = round((time.perf_counter() - validation_started) * 1000.0, 3)

            algorithm_started = time.perf_counter()
            result = self.algorithm.evaluate(runtime_result)
            timing["algorithm_ms"] = round((time.perf_counter() - algorithm_started) * 1000.0, 3)

            try:
                capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
            except (TypeError, ValueError, OverflowError):
                capture_timestamp_ms = 0
            capture_timestamp = (
                capture_timestamp_ms / 1000.0 if capture_timestamp_ms > 0 else packet.started_timestamp
            )

            robot_started = time.perf_counter()
            interim_latency_ms = (time.monotonic() - packet.started_monotonic) * 1000.0
            robot = self._robot_message(
                packet.frame_id,
                capture_timestamp,
                result,
                interim_latency_ms,
                packet.request_id,
                packet.trigger_task_id,
                runtime_result=runtime_result,
            )
            timing["robot_message_build_ms"] = round((time.perf_counter() - robot_started) * 1000.0, 3)

            visualization_started = time.perf_counter()
            diagnostics = {
                "matched_bottle_count": len(
                    [item for item in result.items if item.get("target_type") != "box"]
                ),
                "box_count": len(result.boxes),
                "detected_bottle_count": len(result.bottles),
                "detected_grasp_point_count": len(result.grasp_points),
                "unmatched_bottles": result.unmatched_bottles,
                "unmatched_grasp_points": result.unmatched_grasp_points,
                "ignored_detections": result.ignored,
                "camera_health": {"connected": True, "source": "runtime_inference"},
            }
            # Runtime results are immutable after publication.  A shallow copy
            # avoids recursively duplicating every detection and debug field.
            visualization = dict(runtime_result)
            visualization["detergent_grasp"] = {
                "robot_items": result.items,
                "diagnostics": diagnostics,
                "video_url": self.config["video"]["public_url"],
            }
            producer = self.producer_metadata()
            visualization["producer"] = producer
            timing["visualization_build_ms"] = round(
                (time.perf_counter() - visualization_started) * 1000.0, 3
            )

            decision_started = time.perf_counter()
            decision: Dict[str, Any] = {
                "schema_version": "1.0",
                "message_type": "app_decision",
                "status": "ok",
                "task_id": str(self.config.get("task_id") or "detergent_grasp"),
                "timestamp_ms": _timestamp_ms(),
                "robot_message": robot,
                "visualization_result": visualization,
                "producer": producer,
                "app_timing": timing,
            }
            timing["decision_build_ms"] = round((time.perf_counter() - decision_started) * 1000.0, 3)
            visualization["detergent_grasp"]["app_timing"] = timing

            if dispatch:
                robot["latency_ms"] = round(
                    (time.monotonic() - packet.started_monotonic) * 1000.0, 3
                )
                send_ms, sent = self._dispatch_packet(packet, robot)
                timing["websocket_send_ms"] = send_ms
                timing["websocket_clients_sent"] = sent

            debug_started = time.perf_counter()
            self._save_debug_async(decision)
            timing["debug_enqueue_ms"] = round((time.perf_counter() - debug_started) * 1000.0, 3)
            timing["postprocess_stage_ms"] = round((time.perf_counter() - postprocess_started) * 1000.0, 3)
            final_latency_ms = self.state.success(
                decision, robot, runtime_result, packet.started_monotonic, timing
            )
            robot["latency_ms"] = round(final_latency_ms, 3)
            return decision
        except Exception as raw_error:
            fault_started = time.perf_counter()
            fault_code, fault_type = self._fault_for_error(raw_error)
            timing["fault_mapping_ms"] = round((time.perf_counter() - fault_started) * 1000.0, 3)
            elapsed_ms = (time.monotonic() - packet.started_monotonic) * 1000.0
            robot = self._robot_message(
                packet.frame_id,
                packet.started_timestamp,
                None,
                elapsed_ms,
                packet.request_id,
                packet.trigger_task_id,
                fault_code=fault_code,
                fault_type=fault_type,
            )
            decision = {
                "schema_version": "1.0",
                "message_type": "app_decision",
                "status": "error",
                "task_id": str(self.config.get("task_id") or "detergent_grasp"),
                "timestamp_ms": _timestamp_ms(),
                "robot_message": robot,
                "visualization_result": None,
                "error": {"code": type(raw_error).__name__, "message": str(raw_error)},
                "producer": self.producer_metadata(),
                "app_timing": timing,
            }
            if dispatch:
                robot["latency_ms"] = round(
                    (time.monotonic() - packet.started_monotonic) * 1000.0, 3
                )
                send_ms, sent = self._dispatch_packet(packet, robot)
                timing["websocket_send_ms"] = send_ms
                timing["websocket_clients_sent"] = sent
            debug_started = time.perf_counter()
            self._save_debug_async(decision)
            timing["debug_enqueue_ms"] = round((time.perf_counter() - debug_started) * 1000.0, 3)
            timing["postprocess_stage_ms"] = round((time.perf_counter() - postprocess_started) * 1000.0, 3)
            final_latency_ms = self.state.failure(
                decision, robot, raw_error, packet.started_monotonic, timing
            )
            robot["latency_ms"] = round(final_latency_ms, 3)
            return decision
        finally:
            self.state.end_postprocess()

    def evaluate_once(
        self,
        request_id: Optional[object] = None,
        trigger_task_id: Optional[object] = None,
    ) -> Dict[str, Any]:
        # HTTP/manual requests remain synchronous.  Only Runtime access is
        # serialized with the continuous producer, while CPU postprocess may
        # overlap the next production inference.
        with self.execution_lock:
            packet = self._run_inference_stage(
                request_id=request_id, trigger_task_id=trigger_task_id, continuous=False
            )
            return self._complete_packet(packet, dispatch=False)

    def _enqueue_trigger_packet(self, packet: InferencePacket) -> None:
        # A trigger must not be discarded.  Prefer evicting a queued continuous
        # result; if another trigger is queued, wait until postprocess consumes it.
        while not self.stop_event.is_set():
            try:
                self.result_queue.put_nowait(packet)
                return
            except queue.Full:
                pass
            try:
                previous = self.result_queue.get_nowait()
            except queue.Empty:
                previous = None
            if previous is None:
                continue
            if previous.trigger is None:
                self.state.counters["pipeline_results_dropped"] += 1
                self.state.counters["pipeline_continuous_evicted_for_trigger"] += 1
                continue
            try:
                self.result_queue.put_nowait(previous)
            except queue.Full:
                pass
            self.state.counters["pipeline_trigger_waits"] += 1
            try:
                self.result_queue.put(packet, timeout=0.1)
                return
            except queue.Full:
                continue

    def _enqueue_packet(self, packet: InferencePacket) -> None:
        if packet.trigger is not None:
            self._enqueue_trigger_packet(packet)
            return
        try:
            self.result_queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            previous = self.result_queue.get_nowait()
        except queue.Empty:
            previous = None
        if previous is not None and previous.trigger is not None:
            # Never discard an explicit robot trigger to make room for a
            # continuous frame.  Restore it and drop the new continuous result.
            try:
                self.result_queue.put_nowait(previous)
            except queue.Full:
                pass
            self.state.counters["pipeline_results_dropped"] += 1
            self.state.counters["pipeline_continuous_dropped_for_trigger"] += 1
            return
        if previous is not None:
            self.state.counters["pipeline_results_dropped"] += 1
        try:
            self.result_queue.put_nowait(packet)
        except queue.Full:
            self.state.counters["pipeline_results_dropped"] += 1

    def _inference_loop(self) -> None:
        next_continuous = time.monotonic()
        while not self.stop_event.is_set():
            try:
                trigger = self.trigger_queue.get_nowait()
            except queue.Empty:
                trigger = None
            continuous = self.state.continuous_enabled
            now = time.monotonic()
            due = continuous and now >= next_continuous
            if trigger is not None or due:
                packet = self._run_inference_stage(
                    request_id=trigger.request_id if trigger is not None else None,
                    trigger_task_id=trigger.trigger_task_id if trigger is not None else None,
                    trigger=trigger,
                    continuous=trigger is None,
                )
                if self.pipeline_enabled:
                    self._enqueue_packet(packet)
                else:
                    self._complete_packet(packet, dispatch=True)
                if due:
                    period_s = 1.0 / self.production_fps()
                    next_continuous = max(next_continuous + period_s, time.monotonic())
                continue
            timeout = max(0.005, min(0.1, next_continuous - now)) if continuous else 0.1
            signaled = self.wakeup.wait(timeout)
            self.wakeup.clear()
            if not continuous:
                next_continuous = time.monotonic()
            elif signaled:
                next_continuous = min(next_continuous, time.monotonic())

    def _postprocess_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            age_ms = (time.monotonic() - packet.started_monotonic) * 1000.0
            if packet.trigger is None and age_ms > self.pipeline_max_result_age_ms:
                self.state.counters["pipeline_stale_results_dropped"] += 1
                continue
            self._complete_packet(packet, dispatch=True)

    def _status_message(self) -> Dict[str, Any]:
        snapshot = self.state.snapshot(self.websocket)
        model_name = ""
        runtime_camera_connected = False
        try:
            runtime = self.runtime.status()
            loaded_model = runtime.get("loaded_model") if isinstance(runtime.get("loaded_model"), Mapping) else {}
            model_name = str(loaded_model.get("model_name") or loaded_model.get("model_id") or "")
            runtime_camera_connected = bool(runtime.get("camera_connected"))
        except Exception:
            pass
        camera = self._bridge_health()
        fault_code = FAULT_NONE if camera["connected"] else FAULT_CAMERA_DISCONNECTED
        fault_type = FAULT_TYPE_NONE if camera["connected"] else FAULT_TYPE_CAMERA_DISCONNECTED
        return {
            "type": "status",
            "online": True,
            "fps": snapshot["detection_fps"],
            "configured_fps": snapshot["configured_detection_fps"],
            "model": model_name,
            "camera_connected": bool(camera["connected"]),
            "runtime_camera_connected": runtime_camera_connected,
            "last_color_age_ms": camera["last_color_age_ms"],
            "fault_code": fault_code,
            "fault_type": fault_type,
            "latency_ms": snapshot["last_latency_ms"],
            "continuous_enabled": snapshot["continuous_enabled"],
            "clients": snapshot["websocket"]["clients"],
            "video_url": snapshot["video"]["url"],
            "push_mode": "every_completed_result",
        }

    def _status_loop(self) -> None:
        interval = max(0.5, float(self.config["websocket"].get("status_interval_s", 2.0)))
        while not self.stop_event.wait(interval):
            if self.websocket.client_count() > 0:
                self.websocket.broadcast_json(self._status_message())

    def _save_debug_async(self, decision: Mapping[str, Any]) -> None:
        if not self.debug_enabled:
            return
        snapshot = deepcopy(dict(decision))
        threading.Thread(
            target=self._save_debug,
            args=(snapshot,),
            name="detergent-grasp-debug-writer",
            daemon=True,
        ).start()

    def _save_debug(self, decision: Mapping[str, Any]) -> None:
        with self.debug_lock:
            self.debug_root.mkdir(parents=True, exist_ok=True)
            (self.debug_root / "result.json").write_text(
                json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            visualization = decision.get("visualization_result")
            if not isinstance(visualization, Mapping):
                return
            try:
                rgb = self.runtime.snapshot()
                if rgb:
                    (self.debug_root / "rgb.jpg").write_bytes(rgb)
                    self._draw_overlay(rgb, visualization, self.debug_root / "overlay.jpg")
            except Exception:
                return

    @staticmethod
    def _draw_overlay(rgb_bytes: bytes, visualization: Mapping[str, Any], output_path: Path) -> None:
        image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        metadata = visualization.get("detergent_grasp") if isinstance(visualization.get("detergent_grasp"), Mapping) else {}
        items = metadata.get("robot_items") if isinstance(metadata.get("robot_items"), list) else []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            points = item.get("obb_points")
            if isinstance(points, list) and len(points) >= 4:
                polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [polygon], True, (0, 180, 255), 2)
            center = item.get("center_px")
            if isinstance(center, list) and len(center) >= 2:
                point = (int(round(float(center[0]))), int(round(float(center[1]))))
                cv2.circle(image, point, 5, (0, 0, 255), -1)
                label = "{} {:.2f} {:.1f}deg".format(
                    item.get("target_type") or item.get("class_id"),
                    float(item.get("confidence") or 0.0),
                    float(item.get("angle_deg") or 0.0),
                )
                cv2.putText(
                    image,
                    label,
                    (point[0] + 6, max(18, point[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 180, 255),
                    1,
                )
        cv2.imwrite(str(output_path), image)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsDetergentGrasp/1.0"

    @property
    def service(self) -> DetergentGraspVisionService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, code: int, document: Mapping[str, Any]) -> None:
        body = _json_bytes(document)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if size < 0 or size > MAX_HTTP_BODY:
            raise ValueError("request body exceeds size limit")
        raw = self.rfile.read(size) if size else b"{}"
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be a JSON object") from error
        if not isinstance(document, dict):
            raise ValueError("request JSON root must be object")
        return document

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        snapshot = self.service.state.snapshot(self.service.websocket)
        if path == "/health":
            status = self.service._status_message()
            self._send(
                200,
                {
                    "schema_version": "1.0",
                    "message_type": "app_health",
                    "status": "ok",
                    "health": "ok" if status["camera_connected"] else "degraded",
                    "app_id": "detergent_grasp_vision",
                    "camera_connected": status["camera_connected"],
                    "fault_code": status["fault_code"],
                    "fault_type": status["fault_type"],
                    "timestamp_ms": _timestamp_ms(),
                },
            )
        elif path in {"/api/app/status", "/api/gateway/status", "/api/ws/status"}:
            snapshot["external_status"] = self.service._status_message()
            snapshot["inference_settings"] = self.service.inference_settings()
            snapshot["pipeline"] = self.service.pipeline_status()
            self._send(200, snapshot)
        elif path == "/api/ws/clients":
            self._send(200, {"status": "ok", "clients": self.service.websocket.client_snapshot()})
        elif path in {"/api/app/registers", "/api/gateway/registers"}:
            self._send(
                200,
                {
                    "schema_version": "1.0",
                    "message_type": "register_snapshot",
                    "status": "not_applicable",
                    "protocol": "websocket",
                    "registers": [],
                },
            )
        elif path == "/api/app/latest_decision":
            latest = snapshot.get("latest_decision")
            if isinstance(latest, Mapping):
                latest = deepcopy(dict(latest))
                latest["producer"] = self.service.producer_metadata()
                visualization = latest.get("visualization_result")
                if isinstance(visualization, dict):
                    visualization["producer"] = self.service.producer_metadata()
            self._send(200, latest or {"status": "empty", "message_type": "app_decision"})
        elif path == "/api/app/latest_gateway_message":
            self._send(200, snapshot.get("latest_gateway_message") or {"status": "empty", "type": "detection", "items": []})
        elif path == "/api/app/inference_settings":
            self._send(200, self.service.inference_settings())
        else:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        allowed = {
            "/api/app/evaluate_once",
            "/api/task/evaluate_once",
            "/api/app/trigger",
            "/api/app/inference_settings",
        }
        if path not in allowed:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})
            return
        try:
            document = self._read_json()
            if path == "/api/app/inference_settings":
                self._send(200, self.service.set_production_fps(document.get("detection_fps")))
                return
            request_id = document.get("request_id")
            trigger_task_id = document.get("task_id") if path == "/api/app/trigger" else None
            if path == "/api/app/trigger" and trigger_task_id is not None and not self.service._trigger_task_allowed(trigger_task_id):
                raise ValueError("unsupported task_id: {}".format(trigger_task_id))
            if request_id is None:
                self.service.manual_request_id += 1
                request_id = "manual-{}".format(self.service.manual_request_id)
            decision = self.service.evaluate_once(request_id, trigger_task_id)
            self._send(200, decision)
        except ValueError as error:
            self._send(400, {"status": "error", "error": {"code": "INVALID_REQUEST", "message": str(error)}})
        except Exception as error:
            self._send(500, {"status": "error", "error": {"code": type(error).__name__, "message": str(error)}})


def run(config: Mapping[str, Any]) -> int:
    service = DetergentGraspVisionService(config)
    app = config["app"]
    server = ReusableThreadingHTTPServer((str(app["listen_host"]), int(app["listen_port"])), StatusHandler)
    server.service = service  # type: ignore[attr-defined]
    stop_once = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
        service.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    service.start()
    thread = threading.Thread(target=server.serve_forever, name="detergent-grasp-http", daemon=True)
    thread.start()
    ws = config["websocket"]
    print(
        "Detergent Grasp Vision started: ws={}:{}{} http={}:{} runtime={} video={}".format(
            ws["listen_host"],
            ws["listen_port"],
            ws["path"],
            app["listen_host"],
            app["listen_port"],
            config["runtime"]["url"],
            config["video"]["public_url"],
        )
    )
    try:
        while not stop_once.wait(1.0):
            pass
    finally:
        service.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps detergent grasp WebSocket/HTTP app")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="detergent_grasp YAML")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
