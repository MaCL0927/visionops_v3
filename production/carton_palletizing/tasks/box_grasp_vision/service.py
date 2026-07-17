#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP + WebSocket service for segmentation-based carton grasp geometry."""
from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_palletizing.config import DEFAULT_CONFIG_PATH, load_config
from production.carton_palletizing.tasks.box_grasp_vision.algorithm import BoxGraspAlgorithm
from production.carton_palletizing.tasks.box_grasp_vision.websocket_server import WebSocketJsonServer, WebSocketSession

MAX_HTTP_BODY = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
FAULT_NONE = 0
FAULT_CAMERA_DISCONNECTED = 3101
FAULT_VISION_INFERENCE_ERROR = 3201
FAULT_TYPE_NONE = "NONE"
FAULT_TYPE_CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
FAULT_TYPE_VISION_INFERENCE_ERROR = "VISION_INFERENCE_ERROR"


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_depth_png(raw: bytes) -> "np.ndarray":
    if not raw:
        raise ValueError("depth image is empty")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("failed to decode depth PNG")
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError("depth image shape is invalid: {}".format(image.shape))
    return image.astype(np.uint16, copy=False)


class UpstreamError(ConnectionError):
    pass


class CameraUnavailableError(UpstreamError):
    pass


class JsonHttpClient:
    def __init__(self, timeout_s: float = 5.0, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)

    def request_bytes(self, method: str, url: str, body: Optional[bytes] = None) -> bytes:
        headers = {"Accept": "application/json,image/jpeg,image/png,*/*", "User-Agent": "visionops-box-grasp/1.0"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise UpstreamError("{} {} HTTP {}: {}".format(method, url, error.code, detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise UpstreamError("{} {} failed: {}".format(method, url, getattr(error, "reason", error))) from error
        if len(raw) > self.max_response_bytes:
            raise UpstreamError("upstream response exceeds size limit")
        return raw

    def request_json(self, method: str, url: str, document: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        body = None
        if document is not None:
            body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        raw = self.request_bytes(method, url, body)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("upstream returned non-JSON content") from error
        if not isinstance(payload, dict):
            raise UpstreamError("upstream JSON root must be an object")
        return payload


class RuntimeClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.http = JsonHttpClient(timeout_s)

    def infer_once(self) -> Dict[str, Any]:
        result = self.http.request_json("POST", self.base_url + "/api/runtime/infer_once", {})
        if result.get("message_type") != "inference_result" or result.get("status") != "ok":
            raise UpstreamError("Runtime infer_once did not return a successful inference_result")
        return result

    def status(self) -> Dict[str, Any]:
        return self.http.request_json("GET", self.base_url + "/api/runtime/status")

    def snapshot(self) -> bytes:
        return self.http.request_bytes("GET", self.base_url + "/api/runtime/snapshot.jpg")


class CameraBridgeClient:
    def __init__(self, settings: Mapping[str, Any], timeout_s: float, max_depth_age_ms: int) -> None:
        self.base_url = str(settings.get("base_url") or "http://127.0.0.1:18182").rstrip("/")
        self.health_url = self.base_url + str(settings.get("health_path") or "/health")
        self.depth_url = self.base_url + str(settings.get("depth_path") or "/stream/depth.png")
        self.deproject_url = self.base_url + str(settings.get("deproject_path") or "/api/coordinate/deproject")
        self.http = JsonHttpClient(timeout_s)
        self.max_depth_age_ms = max(0, int(max_depth_age_ms))

    def health(self) -> Dict[str, Any]:
        try:
            return self.http.request_json("GET", self.health_url)
        except UpstreamError as error:
            raise CameraUnavailableError("camera bridge health unavailable: {}".format(error)) from error

    @staticmethod
    def _age(document: Mapping[str, Any], name: str) -> int:
        try:
            return int(document.get(name, -1))
        except (TypeError, ValueError, OverflowError):
            return -1

    def require_ready(self, need_depth: bool) -> Dict[str, Any]:
        health = self.health()
        color_age = self._age(health, "last_color_age_ms")
        depth_age = self._age(health, "last_depth_age_ms")
        camera_connected = health.get("camera_connected")
        started = health.get("camera_started")
        if camera_connected is False or (camera_connected is None and started is not True):
            raise CameraUnavailableError("camera bridge reports camera disconnected")
        if self.max_depth_age_ms > 0 and color_age >= 0 and color_age > self.max_depth_age_ms:
            raise CameraUnavailableError("RGB frame is stale: {}ms".format(color_age))
        if need_depth and self.max_depth_age_ms > 0 and (depth_age < 0 or depth_age > self.max_depth_age_ms):
            raise CameraUnavailableError("depth frame is stale: {}ms".format(depth_age))
        return health

    def depth(self, health: Optional[Mapping[str, Any]] = None) -> Tuple["np.ndarray", bytes, Dict[str, Any]]:
        current = dict(health) if isinstance(health, Mapping) else self.require_ready(True)
        age = self._age(current, "last_depth_age_ms")
        if self.max_depth_age_ms > 0 and (age < 0 or age > self.max_depth_age_ms):
            raise CameraUnavailableError("depth frame is stale: {}ms".format(age))
        try:
            raw = self.http.request_bytes("GET", self.depth_url)
            depth = _decode_depth_png(raw)
        except (UpstreamError, ValueError) as error:
            raise CameraUnavailableError("camera depth unavailable: {}".format(error)) from error
        return depth, raw, current

    def deproject(self, points: Sequence[Sequence[float]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        response = self.http.request_json("POST", self.deproject_url, {"points": [list(point[:3]) for point in points]})
        if response.get("ok") is not True:
            raise UpstreamError("camera SDK deprojection failed: {}".format(response.get("error") or "unknown"))
        raw_points = response.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != len(points):
            raise UpstreamError("camera SDK deprojection result count mismatch")
        output = []  # type: List[List[float]]
        for item in raw_points:
            position = item.get("position_camera") if isinstance(item, Mapping) else None
            if not isinstance(position, list) or len(position) < 3 or item.get("valid") is not True:
                output.append([0.0, 0.0, 0.0])
                continue
            try:
                output.append([float(position[0]), float(position[1]), float(position[2])])
            except (TypeError, ValueError, OverflowError):
                output.append([0.0, 0.0, 0.0])
        return output, response


@dataclass(frozen=True)
class TriggerRequest:
    session: WebSocketSession
    request_id: object


class ServiceState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.frame_id = 0
        self.busy = False
        self.continuous_enabled = bool(config["box_grasp"]["websocket"].get("auto_start", True))
        self.latest_decision = None  # type: Optional[Dict[str, Any]]
        self.latest_robot_message = None  # type: Optional[Dict[str, Any]]
        self.latest_runtime_result = None  # type: Optional[Dict[str, Any]]
        self.last_error = None  # type: Optional[Dict[str, Any]]
        self.last_latency_ms = 0.0
        self.counters = defaultdict(int)  # type: Dict[str, int]
        self.inference_times = deque(maxlen=100)  # type: deque

    def next_frame_id(self) -> int:
        with self.lock:
            self.frame_id += 1
            return self.frame_id

    def set_continuous(self, enabled: bool) -> None:
        with self.lock:
            self.continuous_enabled = bool(enabled)

    def begin(self) -> None:
        with self.lock:
            self.busy = True
            self.counters["inference_requests"] += 1

    def success(self, decision: Mapping[str, Any], robot_message: Mapping[str, Any], runtime_result: Mapping[str, Any], latency_ms: float) -> None:
        with self.lock:
            self.busy = False
            self.latest_decision = deepcopy(dict(decision))
            self.latest_robot_message = deepcopy(dict(robot_message))
            self.latest_runtime_result = deepcopy(dict(runtime_result))
            self.last_error = None
            self.last_latency_ms = float(latency_ms)
            self.inference_times.append(time.monotonic())
            self.counters["inference_success"] += 1

    def failure(self, decision: Mapping[str, Any], robot_message: Mapping[str, Any], error: Exception, latency_ms: float) -> None:
        with self.lock:
            self.busy = False
            self.latest_decision = deepcopy(dict(decision))
            self.latest_robot_message = deepcopy(dict(robot_message))
            self.last_latency_ms = float(latency_ms)
            self.last_error = {"code": type(error).__name__, "message": str(error), "timestamp_ms": _timestamp_ms()}
            self.counters["inference_failure"] += 1

    def fps(self) -> float:
        with self.lock:
            times = list(self.inference_times)
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return round((len(times) - 1) / elapsed, 3) if elapsed > 0 else 0.0

    def snapshot(self, websocket: Optional[WebSocketJsonServer] = None) -> Dict[str, Any]:
        ws = self.config["box_grasp"]["websocket"]
        with self.lock:
            return {
                "schema_version": "1.0",
                "message_type": "app_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "app_id": "box_grasp_vision",
                "app_instance_id": "carton-palletizing-box-grasp",
                "component": self.config["box_grasp"]["component"],
                "device_id": self.config["box_grasp"]["device_id"],
                "timestamp_ms": _timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "continuous_enabled": self.continuous_enabled,
                "detection_fps": self.fps(),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "websocket": {
                    "listen_host": ws["listen_host"],
                    "listen_port": ws["listen_port"],
                    "path": ws["path"],
                    "clients": websocket.client_count() if websocket else 0,
                },
                "video": {"type": "mjpeg", "url": self.config["box_grasp"]["video"]["public_url"], "sync": "soft"},
                "runtime_url": self.config["box_grasp"]["runtime"]["url"],
                "latest_decision": deepcopy(self.latest_decision),
                "latest_gateway_message": deepcopy(self.latest_robot_message),
                "register_snapshot": [],
                "counters": dict(self.counters),
                "last_error": deepcopy(self.last_error),
            }


class BoxGraspVisionService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.settings = config["box_grasp"]
        timeout_s = float(self.settings["app"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeClient(str(self.settings["runtime"]["url"]), timeout_s)
        depth_settings = self.settings["algorithm"]["depth"]
        self.algorithm = BoxGraspAlgorithm(self.settings["algorithm"])
        self.bridge = CameraBridgeClient(config["camera_bridge"], timeout_s, int(depth_settings.get("max_age_ms", 1500)))
        self.state = ServiceState(config)
        self.execution_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.manual_request_id = 0
        self.trigger_queue = queue.Queue(maxsize=int(self.settings["websocket"].get("trigger_queue_size", 32)))
        self.worker_thread = None  # type: Optional[threading.Thread]
        self.status_thread = None  # type: Optional[threading.Thread]
        self.debug_lock = threading.Lock()
        debug = self.settings.get("debug") if isinstance(self.settings.get("debug"), Mapping) else {}
        self.debug_enabled = bool(debug.get("save_every_trigger", True))
        self.debug_root = Path(str(debug.get("save_root", "/tmp/visionops_v3/carton_palletizing/box_grasp_vision/latest")))
        ws = self.settings["websocket"]
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

    def start(self) -> None:
        self.websocket.start()
        self.worker_thread = threading.Thread(target=self._worker_loop, name="box-grasp-inference", daemon=True)
        self.status_thread = threading.Thread(target=self._status_loop, name="box-grasp-status", daemon=True)
        self.worker_thread.start()
        self.status_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wakeup.set()
        self.websocket.stop()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        if self.status_thread is not None:
            self.status_thread.join(timeout=3.0)

    @staticmethod
    def _valid_request_id(value: object) -> bool:
        return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) != ""

    def _ack(self, session: WebSocketSession, request_type: str, success: bool, request_id: object = None, **extra: Any) -> None:
        document = {"type": "ack", "request_type": request_type, "success": bool(success), "timestamp": time.time()}
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
        self.wakeup.set()

    def _on_ws_json(self, session: WebSocketSession, document: Dict[str, Any]) -> None:
        message_type = str(document.get("type") or "")
        if message_type == "control":
            command = str(document.get("command") or "").lower()
            request_id = document.get("request_id")
            if command in {"start", "stop"}:
                self.state.set_continuous(command == "start")
                self._ack(session, "control", True, request_id, command=command)
                self.wakeup.set()
                return
            if command == "trigger":
                if not self._valid_request_id(request_id):
                    self._ack(session, "control", False, request_id, command=command, error="trigger requires non-empty request_id")
                    return
                try:
                    self.trigger_queue.put_nowait(TriggerRequest(session, request_id))
                except queue.Full:
                    self._ack(session, "control", False, request_id, command=command, error="trigger queue full")
                    return
                self._ack(session, "control", True, request_id, command=command, queued=True)
                self.wakeup.set()
                return
            self._ack(session, "control", False, request_id, command=command, error="unsupported command")
            return
        if message_type == "ping":
            session.send_json({"type": "pong", "timestamp": time.time()})
            return
        self._ack(session, message_type or "unknown", False, document.get("request_id"), error="unsupported message type")

    def _validate_runtime(self, result: Mapping[str, Any]) -> None:
        runtime = self.settings["runtime"]
        task_type = str(result.get("task_type") or "").strip().lower()
        accepted = {str(item).strip().lower() for item in runtime.get("accepted_task_types", [])}
        if accepted and task_type not in accepted:
            raise ValueError("box grasp Runtime must load segmentation model; task_type={!r}, accepted={}".format(task_type, sorted(accepted)))

    @staticmethod
    def _external_fault(camera_connected: bool, inference_error: bool = False) -> Tuple[int, str]:
        if not camera_connected:
            return FAULT_CAMERA_DISCONNECTED, FAULT_TYPE_CAMERA_DISCONNECTED
        if inference_error:
            return FAULT_VISION_INFERENCE_ERROR, FAULT_TYPE_VISION_INFERENCE_ERROR
        return FAULT_NONE, FAULT_TYPE_NONE

    @staticmethod
    def _protocol_point(value: object, dimensions: int) -> List[float]:
        if not isinstance(value, (list, tuple)) or len(value) < dimensions:
            return [0.0 for _ in range(dimensions)]
        output = []  # type: List[float]
        for index in range(dimensions):
            try:
                output.append(round(float(value[index]), 3))
            except (TypeError, ValueError, OverflowError):
                output.append(0.0)
        return output

    @classmethod
    def _build_grasp_point_items(cls, item: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Convert one detected carton into two robot-facing grasp-point items.

        The unified robot contract treats every element of ``items`` as one
        grasp point rather than one detected product.  Both points from the
        same carton therefore share ``id``, ``class_id`` and ``confidence``;
        only ``center_px`` and ``position_camera`` differ.
        """
        grasp_px = item.get("grasp_points_px") if isinstance(item.get("grasp_points_px"), Mapping) else {}
        grasp_camera = item.get("grasp_points_camera") if isinstance(item.get("grasp_points_camera"), Mapping) else {}

        point_pairs = [
            (
                cls._protocol_point(grasp_px.get("left_mid"), 2),
                cls._protocol_point(grasp_camera.get("left_mid"), 3),
            ),
            (
                cls._protocol_point(grasp_px.get("right_mid"), 2),
                cls._protocol_point(grasp_camera.get("right_mid"), 3),
            ),
        ]
        # Keep output deterministic while avoiding left/right-specific fields.
        # The robot may group by id and distinguish the two points by pixel x.
        point_pairs.sort(key=lambda pair: (pair[0][0], pair[0][1]))

        try:
            item_id = int(item.get("id", 0))
        except (TypeError, ValueError, OverflowError):
            item_id = 0
        try:
            class_id = int(item.get("class_id", 0))
        except (TypeError, ValueError, OverflowError):
            class_id = 0
        try:
            confidence = round(float(item.get("confidence", 0.0)), 6)
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0

        return [
            {
                "id": item_id,
                "class_id": class_id,
                "confidence": confidence,
                "position_camera": position_camera,
                "center_px": center_px,
            }
            for center_px, position_camera in point_pairs
        ]

    def _error_result(self, frame_id: int, request_id: object, error: Exception, started_at: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        camera_connected = not isinstance(error, CameraUnavailableError)
        fault_code, fault_type = self._external_fault(camera_connected, inference_error=camera_connected)
        robot = {
            "type": "detection",
            "frame_id": frame_id,
            "timestamp": started_at,
            "items": [],
            "fault_code": fault_code,
            "fault_type": fault_type,
        }
        if request_id is not None:
            robot["request_id"] = request_id
        decision = {
            "schema_version": "1.0",
            "message_type": "app_decision",
            "status": "error",
            "app_id": "box_grasp_vision",
            "task": "segmentation_box_grasp",
            "timestamp_ms": _timestamp_ms(),
            "robot_message": robot,
            "visualization_result": None,
            "error": {"code": type(error).__name__, "message": str(error), "recoverable": True},
        }
        return decision, robot

    def evaluate_once(self, request_id: object = None) -> Dict[str, Any]:
        frame_id = self.state.next_frame_id()
        started_at = time.time()
        started_monotonic = time.monotonic()
        self.state.begin()
        with self.execution_lock:
            try:
                need_depth = bool(self.algorithm.depth_enabled)
                camera_health = self.bridge.require_ready(need_depth)
                runtime_result = self.runtime.infer_once()
                self._validate_runtime(runtime_result)
                classified = self.algorithm.classify(runtime_result)
                depth_bytes = b""
                depth = None
                bridge_debug = {}  # type: Dict[str, Any]
                if classified.items and need_depth:
                    depth, depth_bytes, depth_health = self.bridge.depth(camera_health)
                    bridge_debug["health"] = depth_health

                external_items = []  # type: List[Dict[str, Any]]
                sampled_debug = []  # type: List[Dict[str, Any]]
                for index, item in enumerate(classified.items):
                    if depth is not None:
                        depth_info = self.algorithm.sample_item_depth(item, depth, classified.image_width, classified.image_height)
                        deproject_input = self.algorithm.build_deproject_input(item, depth_info)
                        positions, deproject_debug = self.bridge.deproject(deproject_input)
                    else:
                        depth_info = {name: {"depth_valid": False, "depth_mm": 0, "sample_px": [0, 0], "valid_pixels": 0} for name in self.algorithm.POINT_ORDER}
                        positions = [[0.0, 0.0, 0.0] for _ in self.algorithm.POINT_ORDER]
                        deproject_debug = {"ok": False, "reason": "depth_disabled_or_no_target"}
                    external_items.append(self.algorithm.build_external_item(index, item, depth_info, positions))
                    sampled_debug.append({"source_id": item.get("source_id"), "depth": depth_info, "deproject": deproject_debug})

                latency_ms = (time.monotonic() - started_monotonic) * 1000.0
                try:
                    capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
                except (TypeError, ValueError, OverflowError):
                    capture_timestamp_ms = 0
                timestamp = capture_timestamp_ms / 1000.0 if capture_timestamp_ms > 0 else started_at
                protocol_items = [
                    grasp_point
                    for item in external_items
                    for grasp_point in self._build_grasp_point_items(item)
                ]
                robot = {
                    "type": "detection",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "items": protocol_items,
                    "fault_code": FAULT_NONE,
                    "fault_type": FAULT_TYPE_NONE,
                }
                if request_id is not None:
                    robot["request_id"] = request_id
                visualization = deepcopy(runtime_result)
                visualization["box_grasp"] = {
                    "items": external_items,
                    "point_order": list(self.algorithm.POINT_ORDER),
                    "ignored": classified.ignored,
                }
                decision = {
                    "schema_version": "1.0",
                    "message_type": "app_decision",
                    "status": "ok",
                    "app_id": "box_grasp_vision",
                    "task": "segmentation_box_grasp",
                    "device_id": self.settings["device_id"],
                    "component": self.settings["component"],
                    "timestamp_ms": _timestamp_ms(),
                    "frame_id": runtime_result.get("frame_id"),
                    "result_id": runtime_result.get("result_id"),
                    "robot_message": robot,
                    "visualization_result": visualization,
                }
                self.state.success(decision, robot, runtime_result, latency_ms)
                self._save_debug_async({
                    "decision": decision,
                    "runtime_result": runtime_result,
                    "camera_health": camera_health,
                    "sampled": sampled_debug,
                    "ignored": classified.ignored,
                    "bridge": bridge_debug,
                }, depth_bytes)
                return decision
            except Exception as error:
                latency_ms = (time.monotonic() - started_monotonic) * 1000.0
                decision, robot = self._error_result(frame_id, request_id, error, started_at)
                self.state.failure(decision, robot, error, latency_ms)
                self._save_debug_async({"decision": decision, "error": str(error)}, b"")
                return decision

    def _worker_loop(self) -> None:
        hz = max(0.1, float(self.settings["websocket"].get("detection_hz", 5.0)))
        period_s = 1.0 / hz
        next_continuous = time.monotonic()
        while not self.stop_event.is_set():
            try:
                trigger = self.trigger_queue.get_nowait()
            except queue.Empty:
                trigger = None
            if trigger is not None:
                decision = self.evaluate_once(trigger.request_id)
                robot = decision.get("robot_message") if isinstance(decision.get("robot_message"), Mapping) else {}
                try:
                    trigger.session.send_json(robot)
                except OSError:
                    pass
                continue
            continuous = self.state.continuous_enabled and self.websocket.client_count() > 0
            now = time.monotonic()
            if continuous and now >= next_continuous:
                decision = self.evaluate_once(None)
                robot = decision.get("robot_message") if isinstance(decision.get("robot_message"), Mapping) else {}
                self.websocket.broadcast_json(robot)
                next_continuous = max(next_continuous + period_s, time.monotonic())
                continue
            timeout = max(0.01, min(0.2, next_continuous - now)) if continuous else 0.2
            self.wakeup.wait(timeout)
            self.wakeup.clear()
            if not continuous:
                next_continuous = time.monotonic()

    def _status_message(self) -> Dict[str, Any]:
        snapshot = self.state.snapshot(self.websocket)
        model_name = ""
        camera_connected = False
        try:
            runtime = self.runtime.status()
            loaded_model = runtime.get("loaded_model") if isinstance(runtime.get("loaded_model"), Mapping) else {}
            model_name = str(loaded_model.get("model_name") or loaded_model.get("model_id") or "")
        except Exception:
            pass
        try:
            health = self.bridge.health()
            camera_connected = health.get("camera_connected") is not False and health.get("camera_started") is not False
        except Exception:
            camera_connected = False
        fault_code, fault_type = self._external_fault(camera_connected)
        return {
            "type": "status",
            "task": "box_grasp_vision",
            "online": True,
            "fps": snapshot["detection_fps"],
            "model": model_name,
            "camera_connected": camera_connected,
            "fault_code": fault_code,
            "fault_type": fault_type,
            "latency_ms": snapshot["last_latency_ms"],
            "continuous_enabled": snapshot["continuous_enabled"],
            "clients": snapshot["websocket"]["clients"],
            "video_url": snapshot["video"]["url"],
        }

    def _status_loop(self) -> None:
        interval = max(0.5, float(self.settings["websocket"].get("status_interval_s", 2.0)))
        while not self.stop_event.wait(interval):
            if self.websocket.client_count() > 0:
                self.websocket.broadcast_json(self._status_message())

    def _save_debug_async(self, document: Mapping[str, Any], depth_bytes: bytes) -> None:
        if not self.debug_enabled:
            return
        threading.Thread(target=self._save_debug, args=(deepcopy(dict(document)), bytes(depth_bytes)), name="box-grasp-debug", daemon=True).start()

    def _save_debug(self, document: Mapping[str, Any], depth_bytes: bytes) -> None:
        with self.debug_lock:
            self.debug_root.mkdir(parents=True, exist_ok=True)
            (self.debug_root / "result.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            if depth_bytes:
                (self.debug_root / "depth.png").write_bytes(depth_bytes)
            try:
                rgb = self.runtime.snapshot()
                if rgb:
                    (self.debug_root / "rgb.jpg").write_bytes(rgb)
                    decision = document.get("decision") if isinstance(document.get("decision"), Mapping) else {}
                    visualization = decision.get("visualization_result") if isinstance(decision.get("visualization_result"), Mapping) else {}
                    box_grasp = visualization.get("box_grasp") if isinstance(visualization.get("box_grasp"), Mapping) else {}
                    self._draw_overlay(rgb, box_grasp.get("items"), self.debug_root / "overlay.jpg")
            except Exception:
                pass

    @staticmethod
    def _draw_overlay(rgb: bytes, items_value: object, output: Path) -> None:
        image = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        items = items_value if isinstance(items_value, list) else []
        colors = {"top_left": (255, 0, 0), "top_right": (0, 165, 255), "bottom_right": (0, 0, 255), "bottom_left": (255, 255, 0)}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            contour = np.asarray(item.get("contour_px") or [], dtype=np.int32)
            corners = item.get("corners_px") if isinstance(item.get("corners_px"), Mapping) else {}
            quad = np.asarray([corners.get(name) for name in ("top_left", "top_right", "bottom_right", "bottom_left")], dtype=np.int32)
            if contour.ndim == 2 and contour.shape[0] >= 3:
                cv2.polylines(image, [contour], True, (255, 255, 0), 1)
            if quad.ndim == 2 and quad.shape == (4, 2):
                cv2.polylines(image, [quad], True, (0, 255, 0), 3)
            for name, color in colors.items():
                point = corners.get(name)
                if isinstance(point, list) and len(point) >= 2:
                    x, y = int(round(float(point[0]))), int(round(float(point[1])))
                    cv2.circle(image, (x, y), 7, color, -1)
                    cv2.putText(image, name.upper(), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            center = item.get("center_px")
            grasp = item.get("grasp_points_px") if isinstance(item.get("grasp_points_px"), Mapping) else {}
            for name, point in (("C", center), ("L", grasp.get("left_mid")), ("R", grasp.get("right_mid"))):
                if isinstance(point, list) and len(point) >= 2:
                    x, y = int(round(float(point[0]))), int(round(float(point[1])))
                    cv2.circle(image, (x, y), 7, (0, 255, 255), -1)
                    cv2.putText(image, "{}({},{})".format(name, x, y), (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(output), image)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsBoxGrasp/1.0"

    @property
    def service(self) -> BoxGraspVisionService:
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
            self._send(200, {"schema_version": "1.0", "message_type": "app_health", "status": "ok", "health": snapshot["health"], "app_id": "box_grasp_vision", "timestamp_ms": _timestamp_ms()})
        elif path in {"/api/app/status", "/api/gateway/status", "/api/ws/status"}:
            snapshot["external_status"] = self.service._status_message()
            self._send(200, snapshot)
        elif path == "/api/ws/clients":
            self._send(200, {"status": "ok", "clients": self.service.websocket.client_snapshot()})
        elif path in {"/api/app/registers", "/api/gateway/registers"}:
            self._send(200, {"schema_version": "1.0", "message_type": "register_snapshot", "status": "not_applicable", "protocol": "websocket", "registers": []})
        elif path == "/api/app/latest_decision":
            self._send(200, snapshot.get("latest_decision") or {"status": "empty", "message_type": "app_decision"})
        elif path == "/api/app/latest_gateway_message":
            self._send(200, snapshot.get("latest_gateway_message") or {"status": "empty", "type": "detection", "items": []})
        else:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/api/app/evaluate_once", "/api/task/evaluate_once"}:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})
            return
        try:
            document = self._read_json()
            request_id = document.get("request_id")
            if request_id is None:
                self.service.manual_request_id += 1
                request_id = "manual-{}".format(self.service.manual_request_id)
            self._send(200, self.service.evaluate_once(request_id))
        except Exception as error:
            self._send(500, {"status": "error", "error": {"code": type(error).__name__, "message": str(error)}})


def run(config: Mapping[str, Any]) -> int:
    service = BoxGraspVisionService(config)
    http = config["box_grasp"]["app"]
    server = ReusableThreadingHTTPServer((str(http["listen_host"]), int(http["listen_port"])), StatusHandler)
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
    thread = threading.Thread(target=server.serve_forever, name="box-grasp-http", daemon=True)
    thread.start()
    ws = config["box_grasp"]["websocket"]
    print(
        "Carton Box Grasp Vision started: ws={}:{}{} http={}:{} runtime={} video={}".format(
            ws["listen_host"], ws["listen_port"], ws["path"], http["listen_host"], http["listen_port"],
            config["box_grasp"]["runtime"]["url"], config["box_grasp"]["video"]["public_url"]
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Segmentation carton corner/grasp-point WebSocket service")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
