#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP + WebSocket service for segmentation-based carton grasp geometry."""
from __future__ import annotations

import argparse
import json
import math
import os
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

from production.carton_bundle_grasp.config import DEFAULT_CONFIG_PATH, load_config
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.algorithm import CartonBundleGraspAlgorithm, GeometryError
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.local_ipc import RawLocalHttpClient, SharedDepthReader
from production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.websocket_server import WebSocketJsonServer, WebSocketSession

MAX_HTTP_BODY = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
FAULT_NONE = 0
FAULT_CAMERA_DISCONNECTED = 3101
FAULT_VISION_INFERENCE_ERROR = 3201
FAULT_TYPE_NONE = "NONE"
FAULT_TYPE_CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
FAULT_TYPE_VISION_INFERENCE_ERROR = "VISION_INFERENCE_ERROR"

# The production App owns one authoritative target FPS.  WebSocket clients and
# Collector production mode receive every completed result; there is no separate
# WebSocket push frequency.  A persisted v2 setting written through the App API
# overrides this startup default.  Legacy v1 files are intentionally ignored so
# an old hard-coded 5 Hz value cannot silently throttle a newly upgraded service.
DEFAULT_PRODUCTION_INFERENCE_FPS = 15.0
MIN_PRODUCTION_INFERENCE_FPS = 0.1
MAX_PRODUCTION_INFERENCE_FPS = 30.0
INFERENCE_SETTINGS_SCHEMA_VERSION = "2.0"


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


@dataclass(frozen=True)
class HttpBytesResult:
    body: bytes
    status_code: int
    headers: Mapping[str, str]
    headers_wait_ms: float
    body_read_ms: float
    total_ms: float
    connect_ms: float = 0.0
    send_ms: float = 0.0
    transport: str = "urllib"

    def header_float(self, name: str) -> float:
        raw = self.headers.get(name.lower())
        try:
            return float(raw) if raw is not None else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0


class JsonHttpClient:
    def __init__(
        self,
        timeout_s: float = 5.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        raw_local_enabled: bool = True,
        fallback_urllib: bool = True,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)
        self.raw_local_enabled = bool(raw_local_enabled)
        self.fallback_urllib = bool(fallback_urllib)
        self.raw_client = RawLocalHttpClient(self.timeout_s, self.max_response_bytes)
        self.stats_lock = threading.Lock()
        self.raw_request_count = 0
        self.raw_failure_count = 0
        self.urllib_request_count = 0
        self.last_transport = "none"
        self.last_raw_error = ""

    def status(self) -> Dict[str, Any]:
        with self.stats_lock:
            return {
                "raw_local_enabled": self.raw_local_enabled,
                "fallback_urllib": self.fallback_urllib,
                "raw_request_count": self.raw_request_count,
                "raw_failure_count": self.raw_failure_count,
                "urllib_request_count": self.urllib_request_count,
                "last_transport": self.last_transport,
                "last_raw_error": self.last_raw_error,
            }

    def _record_transport(self, transport: str, raw_error: str = "") -> None:
        with self.stats_lock:
            self.last_transport = transport
            if transport == "raw_socket":
                self.raw_request_count += 1
                self.last_raw_error = ""
            elif transport == "urllib":
                self.urllib_request_count += 1
                if raw_error:
                    self.last_raw_error = raw_error
            elif transport == "raw_error":
                self.raw_failure_count += 1
                self.last_raw_error = raw_error

    def request_bytes_timed(self, method: str, url: str, body: Optional[bytes] = None) -> HttpBytesResult:
        if self.raw_local_enabled and self.raw_client.supports(url):
            try:
                response = self.raw_client.request(method, url, body)
                if response.status_code >= 400:
                    detail = response.body[:1000].decode("utf-8", errors="replace")
                    raise UpstreamError("{} {} HTTP {}: {}".format(method, url, response.status_code, detail))
                self._record_transport("raw_socket")
                return HttpBytesResult(
                    body=response.body,
                    status_code=response.status_code,
                    headers=response.headers,
                    headers_wait_ms=response.headers_wait_ms,
                    body_read_ms=response.body_read_ms,
                    total_ms=response.total_ms,
                    connect_ms=response.connect_ms,
                    send_ms=response.send_ms,
                    transport=response.transport,
                )
            except (OSError, ValueError, ConnectionError, TimeoutError) as error:
                self._record_transport("raw_error", str(error))
                if not self.fallback_urllib:
                    raise UpstreamError("{} {} raw HTTP failed: {}".format(method, url, error)) from error
        headers = {"Accept": "application/json,image/jpeg,image/png,*/*", "User-Agent": "visionops-carton-bundle/1.0"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                headers_received = time.perf_counter()
                raw = response.read(self.max_response_bytes + 1)
                finished = time.perf_counter()
                status_code = int(getattr(response, "status", 200))
                normalized_headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise UpstreamError("{} {} HTTP {}: {}".format(method, url, error.code, detail)) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise UpstreamError("{} {} failed: {}".format(method, url, getattr(error, "reason", error))) from error
        if len(raw) > self.max_response_bytes:
            raise UpstreamError("upstream response exceeds size limit")
        self._record_transport("urllib")
        return HttpBytesResult(
            body=raw,
            status_code=status_code,
            headers=normalized_headers,
            headers_wait_ms=(headers_received - started) * 1000.0,
            body_read_ms=(finished - headers_received) * 1000.0,
            total_ms=(finished - started) * 1000.0,
            transport="urllib",
        )

    def request_bytes(self, method: str, url: str, body: Optional[bytes] = None) -> bytes:
        return self.request_bytes_timed(method, url, body).body

    @staticmethod
    def decode_json(raw: bytes) -> Dict[str, Any]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("upstream returned non-JSON content") from error
        if not isinstance(payload, dict):
            raise UpstreamError("upstream JSON root must be an object")
        return payload

    def request_json(self, method: str, url: str, document: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        body = None
        if document is not None:
            body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self.request_bytes_timed(method, url, body)
        return self.decode_json(response.body)


class RuntimeClient:
    def __init__(self, base_url: str, timeout_s: float, ipc_settings: Mapping[str, Any]) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.http = JsonHttpClient(
            timeout_s,
            raw_local_enabled=bool(ipc_settings.get("raw_http_enabled", True)),
            fallback_urllib=bool(ipc_settings.get("raw_http_fallback_urllib", True)),
        )

    @staticmethod
    def decode_inference(raw: bytes) -> Dict[str, Any]:
        result = JsonHttpClient.decode_json(raw)
        if result.get("message_type") != "inference_result" or result.get("status") != "ok":
            raise UpstreamError("Runtime infer_once did not return a successful inference_result")
        return result

    def infer_once_raw(self) -> HttpBytesResult:
        body = b"{}"
        return self.http.request_bytes_timed(
            "POST",
            self.base_url + "/api/runtime/infer_once",
            body,
        )

    def infer_once(self) -> Dict[str, Any]:
        return self.decode_inference(self.infer_once_raw().body)

    def status(self) -> Dict[str, Any]:
        return self.http.request_json("GET", self.base_url + "/api/runtime/status")

    def snapshot(self) -> bytes:
        return self.http.request_bytes("GET", self.base_url + "/api/runtime/snapshot.jpg")

    def transport_status(self) -> Dict[str, Any]:
        status = self.http.status()
        status["base_url"] = self.base_url
        return status


class CameraBridgeClient:
    def __init__(
        self,
        settings: Mapping[str, Any],
        timeout_s: float,
        max_depth_age_ms: int,
        ipc_settings: Mapping[str, Any],
    ) -> None:
        self.base_url = str(settings.get("base_url") or "http://127.0.0.1:18182").rstrip("/")
        self.health_url = self.base_url + str(settings.get("health_path") or "/health")
        self.depth_url = self.base_url + str(settings.get("depth_path") or "/stream/depth.png")
        self.deproject_url = self.base_url + str(settings.get("deproject_path") or "/api/coordinate/deproject")
        self.sample_deproject_url = self.base_url + str(
            settings.get("sample_deproject_path") or "/api/coordinate/sample_deproject"
        )
        self.http = JsonHttpClient(
            timeout_s,
            raw_local_enabled=bool(ipc_settings.get("raw_http_enabled", True)),
            fallback_urllib=bool(ipc_settings.get("raw_http_fallback_urllib", True)),
        )
        self.max_depth_age_ms = max(0, int(max_depth_age_ms))
        self.shared_depth_enabled = bool(settings.get("shared_depth_enabled", True)) and str(
            settings.get("camera_model") or "orbbec336l"
        ).lower() == "orbbec336l"
        self.shared_depth_fallback_http = bool(settings.get("shared_depth_fallback_http", True))
        self.shared_depth = SharedDepthReader(
            str(settings.get("shared_depth_name") or "/visionops_orbbec336l_depth"),
            self.max_depth_age_ms,
        ) if self.shared_depth_enabled else None

    def transport_status(self) -> Dict[str, Any]:
        status = {
            "http": self.http.status(),
            "base_url": self.base_url,
            "shared_depth_enabled": self.shared_depth is not None,
            "shared_depth_fallback_http": self.shared_depth_fallback_http,
            "shared_depth": self.shared_depth.status() if self.shared_depth is not None else None,
        }
        return status

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

    def sample_deproject(
        self,
        points: Sequence[Sequence[float]],
        image_width: int,
        image_height: int,
        radius_px: int,
        percentile: float,
        min_valid_pixels: int,
        min_depth_mm: int,
        max_depth_mm: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Sample D2C depth and deproject points in one Bridge request.

        Each point is ``[sample_u, sample_v, project_u, project_v]``.  Depth is
        sampled around the inward-shifted sample coordinate, while the resulting
        depth value is projected at the original geometric coordinate.  This
        preserves the previous box-edge behaviour without transferring a full
        16-bit depth PNG through HTTP for every inference.
        """
        if self.shared_depth is not None:
            try:
                samples, response = self.shared_depth.sample_deproject(
                    points,
                    image_width,
                    image_height,
                    radius_px,
                    percentile,
                    min_valid_pixels,
                    min_depth_mm,
                    max_depth_mm,
                )
                response["_client_timing"] = {
                    "roundtrip_ms": float(response.get("sample_ms") or 0.0),
                    "connect_ms": 0.0,
                    "send_ms": 0.0,
                    "headers_wait_ms": 0.0,
                    "body_read_ms": 0.0,
                    "json_decode_ms": 0.0,
                    "response_bytes": 0,
                    "transport": "posix_shared_memory",
                }
                return samples, response
            except (OSError, ValueError, RuntimeError) as error:
                self.shared_depth.last_error = str(error)
                if not self.shared_depth_fallback_http:
                    raise CameraUnavailableError("shared depth unavailable: {}".format(error)) from error

        document = {
            "points": [list(point[:4]) for point in points],
            "image_width": int(image_width),
            "image_height": int(image_height),
            "radius_px": int(radius_px),
            "percentile": float(percentile),
            "min_valid_pixels": int(min_valid_pixels),
            "min_depth_mm": int(min_depth_mm),
            "max_depth_mm": int(max_depth_mm),
            "max_depth_age_ms": int(self.max_depth_age_ms),
        }
        body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            timed_response = self.http.request_bytes_timed("POST", self.sample_deproject_url, body)
            decode_started = time.perf_counter()
            response = self.http.decode_json(timed_response.body)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            response["_client_timing"] = {
                "roundtrip_ms": timed_response.total_ms,
                "connect_ms": timed_response.connect_ms,
                "send_ms": timed_response.send_ms,
                "headers_wait_ms": timed_response.headers_wait_ms,
                "body_read_ms": timed_response.body_read_ms,
                "json_decode_ms": decode_ms,
                "response_bytes": len(timed_response.body),
                "transport": timed_response.transport,
            }
        except UpstreamError as error:
            raise CameraUnavailableError("camera depth sample/deproject unavailable: {}".format(error)) from error
        if response.get("ok") is not True:
            raise CameraUnavailableError(
                "camera depth sample/deproject failed: {}".format(response.get("error") or "unknown")
            )
        raw_points = response.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != len(points):
            raise UpstreamError("camera depth sample/deproject result count mismatch")
        output = []  # type: List[Dict[str, Any]]
        for raw in raw_points:
            item = raw if isinstance(raw, Mapping) else {}
            position = item.get("position_camera") if isinstance(item.get("position_camera"), list) else [0, 0, 0]
            if len(position) < 3:
                position = [0, 0, 0]
            try:
                parsed_position = [float(position[0]), float(position[1]), float(position[2])]
            except (TypeError, ValueError, OverflowError):
                parsed_position = [0.0, 0.0, 0.0]
            output.append({
                "depth_valid": bool(item.get("depth_valid")),
                "depth_mm": int(item.get("depth_mm") or 0),
                "sample_px": list(item.get("sample_px") or [0, 0]),
                "valid_pixels": int(item.get("valid_pixels") or 0),
                "position_camera": parsed_position,
                "project_valid": bool(item.get("valid")),
            })

        # The Bridge HTTP response does not expose color intrinsics.  When the
        # M41.2 ROI snapshot path falls back to HTTP, read the shared-depth
        # header only (microsecond-scale) so corner rays remain intrinsics-based
        # and no second 4-point deprojection request is introduced.
        if self.shared_depth is not None and not isinstance(response.get("intrinsics"), Mapping):
            try:
                context = self.shared_depth.read_geometry_context(image_width, image_height)
                response.update(context)
            except (OSError, ValueError, RuntimeError) as error:
                self.shared_depth.last_error = str(error)
        return output, response


@dataclass(frozen=True)
class TriggerRequest:
    session: WebSocketSession
    request_id: object


@dataclass
class InferencePacket:
    frame_id: int
    request_id: object
    started_at: float
    started_monotonic: float
    runtime_raw: bytes = b""
    runtime_result: Optional[Dict[str, Any]] = None
    runtime_lock_wait_ms: float = 0.0
    runtime_http_ms: float = 0.0
    runtime_connect_ms: float = 0.0
    runtime_send_ms: float = 0.0
    runtime_headers_wait_ms: float = 0.0
    runtime_body_read_ms: float = 0.0
    runtime_transport: str = "unknown"
    runtime_json_decode_ms: float = 0.0
    runtime_response_bytes: int = 0
    runtime_server_queue_ms: float = 0.0
    runtime_server_route_ms: float = 0.0
    runtime_internal_ms: float = 0.0
    error: Optional[Exception] = None
    trigger: Optional[TriggerRequest] = None
    continuous: bool = False


class ServiceState:
    def __init__(self, config: Mapping[str, Any], configured_detection_fps: float) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.frame_id = 0
        self.busy = False
        self.inference_busy = False
        self.postprocess_busy = False
        self.continuous_enabled = bool(config["carton_bundle_grasp"]["websocket"].get("auto_start", True))
        self.configured_detection_fps = float(configured_detection_fps)
        self.latest_decision = None  # type: Optional[Dict[str, Any]]
        self.latest_robot_message = None  # type: Optional[Dict[str, Any]]
        self.latest_runtime_result = None  # type: Optional[Dict[str, Any]]
        self.last_error = None  # type: Optional[Dict[str, Any]]
        self.last_latency_ms = 0.0
        self.last_app_timing = {}  # type: Dict[str, Any]
        self.counters = defaultdict(int)  # type: Dict[str, int]
        self.inference_times = deque(maxlen=100)  # type: deque
        self.latency_samples = deque(maxlen=100)  # type: deque
        self.timing_samples = defaultdict(lambda: deque(maxlen=100))

    def next_frame_id(self) -> int:
        with self.lock:
            self.frame_id += 1
            return self.frame_id

    def set_continuous(self, enabled: bool) -> None:
        with self.lock:
            self.continuous_enabled = bool(enabled)

    def set_configured_detection_fps(self, fps: float) -> None:
        with self.lock:
            self.configured_detection_fps = float(fps)

    def begin(self) -> None:
        with self.lock:
            self.busy = True
            self.counters["inference_requests"] += 1

    def begin_inference(self) -> None:
        with self.lock:
            self.inference_busy = True
            self.busy = True
            self.counters["inference_requests"] += 1

    def end_inference(self) -> None:
        with self.lock:
            self.inference_busy = False
            self.busy = self.postprocess_busy

    def begin_postprocess(self) -> None:
        with self.lock:
            self.postprocess_busy = True
            self.busy = True

    def end_postprocess(self) -> None:
        with self.lock:
            self.postprocess_busy = False
            self.busy = self.inference_busy

    def success(
        self,
        decision: Mapping[str, Any],
        robot_message: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        latency_ms: float,
        app_timing: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self.lock:
            self.busy = self.inference_busy or self.postprocess_busy
            # Results are immutable after publication.  Keep references here and
            # deepcopy only when an HTTP/WebSocket snapshot is requested.
            self.latest_decision = dict(decision)
            self.latest_robot_message = dict(robot_message)
            self.latest_runtime_result = dict(runtime_result)
            self.last_error = None
            self.last_latency_ms = float(latency_ms)
            self.last_app_timing = dict(app_timing or {})
            self.inference_times.append(time.monotonic())
            self.latency_samples.append(float(latency_ms))
            for key, value in (app_timing or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.timing_samples[str(key)].append(float(value))
            self.counters["inference_success"] += 1

    def failure(self, decision: Mapping[str, Any], robot_message: Mapping[str, Any], error: Exception, latency_ms: float) -> None:
        with self.lock:
            self.busy = self.inference_busy or self.postprocess_busy
            self.latest_decision = dict(decision)
            self.latest_robot_message = dict(robot_message)
            self.last_latency_ms = float(latency_ms)
            self.latency_samples.append(float(latency_ms))
            self.last_error = {"code": type(error).__name__, "message": str(error), "timestamp_ms": _timestamp_ms()}
            self.counters["inference_failure"] += 1

    def fps(self) -> float:
        with self.lock:
            times = list(self.inference_times)
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return round((len(times) - 1) / elapsed, 3) if elapsed > 0 else 0.0

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * percentile) - 1)))
        return round(ordered[index], 3)

    def snapshot(self, websocket: Optional[WebSocketJsonServer] = None) -> Dict[str, Any]:
        ws = self.config["carton_bundle_grasp"]["websocket"]
        with self.lock:
            latency_values = list(self.latency_samples)
            timing_stats = {
                key: {
                    "p50": self._percentile(list(values), 0.50),
                    "p95": self._percentile(list(values), 0.95),
                }
                for key, values in self.timing_samples.items()
                if values
            }
            return {
                "schema_version": "1.0",
                "message_type": "app_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "app_id": "carton_bundle_grasp_vision",
                "app_instance_id": "carton-palletizing-carton-bundle",
                "component": self.config["carton_bundle_grasp"]["component"],
                "device_id": self.config["carton_bundle_grasp"]["device_id"],
                "timestamp_ms": _timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "inference_busy": self.inference_busy,
                "postprocess_busy": self.postprocess_busy,
                "continuous_enabled": self.continuous_enabled,
                "detection_fps": self.fps(),
                "configured_detection_fps": round(self.configured_detection_fps, 6),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "latency_ms": {
                    "latest": round(self.last_latency_ms, 3),
                    "p50": self._percentile(latency_values, 0.50),
                    "p95": self._percentile(latency_values, 0.95),
                    "samples": len(latency_values),
                },
                "last_app_timing": deepcopy(self.last_app_timing),
                "app_timing_stats": timing_stats,
                "websocket": {
                    "listen_host": ws["listen_host"],
                    "listen_port": ws["listen_port"],
                    "path": ws["path"],
                    "clients": websocket.client_count() if websocket else 0,
                },
                "video": {"type": "mjpeg", "url": self.config["carton_bundle_grasp"]["video"]["public_url"], "sync": "soft"},
                "runtime_url": self.config["carton_bundle_grasp"]["runtime"]["url"],
                "latest_decision": deepcopy(self.latest_decision),
                "latest_gateway_message": deepcopy(self.latest_robot_message),
                "register_snapshot": [],
                "counters": dict(self.counters),
                "last_error": deepcopy(self.last_error),
            }


class CartonBundleGraspVisionService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.settings = config["carton_bundle_grasp"]
        timeout_s = float(self.settings["app"]["request_timeout_ms"]) / 1000.0
        ipc_settings = self.settings.get("ipc") if isinstance(self.settings.get("ipc"), Mapping) else {}
        self.runtime = RuntimeClient(str(self.settings["runtime"]["url"]), timeout_s, ipc_settings)
        depth_settings = self.settings["algorithm"]["depth"]
        self.algorithm = CartonBundleGraspAlgorithm(self.settings["algorithm"])
        self.bridge = CameraBridgeClient(
            config["camera_bridge"],
            timeout_s,
            int(depth_settings.get("max_age_ms", 1500)),
            ipc_settings,
        )
        self.inference_settings_path = Path(
            str(self.settings["app"].get(
                "inference_settings_path",
                "/opt/visionops_v3/configs/runtime/generated/carton_bundle_grasp_inference_settings.json",
            ))
        )
        self.production_fps_lock = threading.Lock()
        self.production_inference_fps = float(self.settings["app"].get("default_production_inference_fps", DEFAULT_PRODUCTION_INFERENCE_FPS))
        self.production_fps_source = "default"
        self._load_production_fps_override()
        self.state = ServiceState(config, self.production_inference_fps)
        self.execution_lock = threading.Lock()
        self.runtime_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.manual_request_id = 0
        self.trigger_queue = queue.Queue(maxsize=int(self.settings["websocket"].get("trigger_queue_size", 32)))
        pipeline_settings = self.settings.get("pipeline") if isinstance(self.settings.get("pipeline"), Mapping) else {}
        self.pipeline_enabled = bool(pipeline_settings.get("enabled", True))
        self.pipeline_max_result_age_ms = max(1, int(pipeline_settings.get("max_result_age_ms", 500)))
        self.result_queue = queue.Queue(maxsize=max(1, int(pipeline_settings.get("result_queue_size", 1))))
        self.worker_thread = None  # type: Optional[threading.Thread]
        self.postprocess_thread = None  # type: Optional[threading.Thread]
        self.status_thread = None  # type: Optional[threading.Thread]
        self.status_cache_lock = threading.Lock()
        self.cached_model_name = ""
        self.cached_camera_connected = False
        self.cached_upstream_status_at = 0.0
        self.debug_lock = threading.Lock()
        debug = self.settings.get("debug") if isinstance(self.settings.get("debug"), Mapping) else {}
        self.debug_enabled = bool(debug.get("save_every_trigger", False))
        self.debug_root = Path(str(debug.get("save_root", "/tmp/visionops_v3/carton_bundle_grasp/latest")))
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

    def _load_production_fps_override(self) -> None:
        try:
            payload = json.loads(self.inference_settings_path.read_text(encoding="utf-8"))
            if str(payload.get("schema_version") or "") != INFERENCE_SETTINGS_SCHEMA_VERSION:
                # Ignore legacy v1 files.  Older frontends could persist the
                # browser polling FPS as 5 Hz and unintentionally throttle the
                # producer after an upgrade.
                return
            hz = float(payload.get("production_inference_fps"))
            if MIN_PRODUCTION_INFERENCE_FPS <= hz <= MAX_PRODUCTION_INFERENCE_FPS:
                self.production_inference_fps = hz
                self.production_fps_source = "persistent"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    def _persist_production_fps(self, hz: float) -> None:
        path = self.inference_settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": INFERENCE_SETTINGS_SCHEMA_VERSION,
                    "production_inference_fps": hz,
                    # Compatibility field for older diagnostic scripts.
                    "detection_fps": hz,
                    "source": "app_inference_settings_api",
                    "updated_at_ms": _timestamp_ms(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))

    def production_fps(self) -> float:
        with self.production_fps_lock:
            return float(self.production_inference_fps)

    def set_production_fps(self, value: object) -> Dict[str, Any]:
        try:
            hz = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("detection_fps 必须是数字") from error
        if not MIN_PRODUCTION_INFERENCE_FPS <= hz <= MAX_PRODUCTION_INFERENCE_FPS:
            raise ValueError("detection_fps 必须位于 0.1..30")
        with self.production_fps_lock:
            self.production_inference_fps = hz
            self.production_fps_source = "persistent"
            self._persist_production_fps(hz)
        self.state.set_configured_detection_fps(hz)
        self.wakeup.set()
        return self.inference_settings()

    # Compatibility aliases for existing tests and external scripts.  They no
    # longer read or write any legacy WebSocket detection_hz setting.
    def detection_hz(self) -> float:
        return self.production_fps()

    def set_detection_hz(self, value: object) -> Dict[str, Any]:
        return self.set_production_fps(value)

    def inference_settings(self) -> Dict[str, Any]:
        configured = self.production_fps()
        return {
            "schema_version": INFERENCE_SETTINGS_SCHEMA_VERSION,
            "message_type": "app_inference_settings",
            "status": "ok",
            "app_id": "carton_bundle_grasp_vision",
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

    def ipc_status(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime.transport_status(),
            "camera_bridge": self.bridge.transport_status(),
        }

    def pipeline_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.pipeline_enabled,
            "result_queue_size": self.result_queue.qsize(),
            "result_queue_capacity": self.result_queue.maxsize,
            "max_result_age_ms": self.pipeline_max_result_age_ms,
            "inference_thread_alive": bool(self.worker_thread and self.worker_thread.is_alive()),
            "postprocess_thread_alive": bool(self.postprocess_thread and self.postprocess_thread.is_alive()),
        }

    def start(self) -> None:
        self.websocket.start()
        self.worker_thread = threading.Thread(target=self._inference_loop, name="carton-bundle-inference", daemon=True)
        self.postprocess_thread = threading.Thread(target=self._postprocess_loop, name="carton-bundle-postprocess", daemon=True)
        self.status_thread = threading.Thread(target=self._status_loop, name="carton-bundle-status", daemon=True)
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

    def _ack(self, session: WebSocketSession, request_type: str, success: bool, request_id: object = None, **extra: Any) -> None:
        document = {"type": "ack", "request_type": request_type, "success": bool(success), "timestamp": time.time()}
        if request_id is not None:
            document["request_id"] = request_id
        document.update(extra)
        session.send_json(document)

    def _on_ws_connect(self, session: WebSocketSession) -> None:
        self.state.counters["connections"] += 1
        try:
            session.send_json(self._status_message(refresh=True))
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
            raise ValueError("carton bundle Runtime must load segmentation model; task_type={!r}, accepted={}".format(task_type, sorted(accepted)))

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
        """Expose the two 525-mm-edge midpoints with the existing robot contract."""
        grasp_px = item.get("grasp_points_px") if isinstance(item.get("grasp_points_px"), Mapping) else {}
        grasp_camera = item.get("grasp_points_camera") if isinstance(item.get("grasp_points_camera"), Mapping) else {}
        roles = item.get("grasp_point_roles") if isinstance(item.get("grasp_point_roles"), list) else []
        if not roles:
            roles = sorted(set(grasp_px.keys()) & set(grasp_camera.keys()))
        point_pairs = []
        for role in roles:
            point_pairs.append((
                cls._protocol_point(grasp_px.get(role), 2),
                cls._protocol_point(grasp_camera.get(role), 3),
            ))
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
            "app_id": "carton_bundle_grasp_vision",
            "task": "segmentation_carton_bundle_grasp",
            "timestamp_ms": _timestamp_ms(),
            "robot_message": robot,
            "visualization_result": None,
            "producer": self.producer_metadata(),
            "error": {"code": type(error).__name__, "message": str(error), "recoverable": True},
        }
        return decision, robot

    @staticmethod
    def _runtime_internal_ms(runtime_result: Mapping[str, Any]) -> float:
        timing = runtime_result.get("timing") if isinstance(runtime_result.get("timing"), Mapping) else {}
        try:
            return float(timing.get("total_ms") or 0.0)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    @staticmethod
    def _point_xy(value: object) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return 0.0, 0.0
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError, OverflowError):
            return 0.0, 0.0

    def _run_inference_stage(
        self,
        request_id: object = None,
        trigger: Optional[TriggerRequest] = None,
        continuous: bool = False,
    ) -> InferencePacket:
        packet = InferencePacket(
            frame_id=self.state.next_frame_id(),
            request_id=request_id,
            started_at=time.time(),
            started_monotonic=time.monotonic(),
            trigger=trigger,
            continuous=continuous,
        )
        self.state.begin_inference()
        lock_started = time.perf_counter()
        try:
            # The C++ Runtime owns one RKNN context. Serialize only the Runtime
            # request. JSON decoding is deliberately deferred to the CPU
            # postprocess stage, so the inference producer can submit the next
            # frame while Python parses/processes the previous result.
            with self.runtime_lock:
                packet.runtime_lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
                response = self.runtime.infer_once_raw()
            packet.runtime_raw = response.body
            packet.runtime_http_ms = response.total_ms
            packet.runtime_connect_ms = response.connect_ms
            packet.runtime_send_ms = response.send_ms
            packet.runtime_headers_wait_ms = response.headers_wait_ms
            packet.runtime_body_read_ms = response.body_read_ms
            packet.runtime_transport = response.transport
            packet.runtime_response_bytes = len(response.body)
            packet.runtime_server_queue_ms = response.header_float("x-visionops-http-queue-ms")
            packet.runtime_server_route_ms = response.header_float("x-visionops-http-route-ms")
        except Exception as error:
            if packet.runtime_lock_wait_ms <= 0.0:
                packet.runtime_lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
            packet.error = error
        finally:
            self.state.end_inference()
        return packet

    def _geometry_for_items(
        self,
        items: Sequence[Mapping[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Recover the physical top plane and fixed-size rectangle per target.

        M41.2 sends only the 96 distributed top-plane samples to the depth path.
        On Orbbec 336L the preferred path takes one stable shared-memory ROI
        snapshot, vectorises the 5x5 depth sampling/deprojection, then derives the
        four corner rays directly from the published color intrinsics.  Final
        corner/grasp pixels therefore never need their own valid depth value.
        """
        if not items or not self.algorithm.depth_enabled:
            return [], [], {
                "mode": "disabled_or_no_target",
                "point_count": 0,
                "plane_point_count": 0,
                "corner_ray_point_count": 0,
                "corner_ray_probe_count": 0,
                "stage_timing_ms": {},
            }

        external_items = []  # type: List[Dict[str, Any]]
        ignored = []  # type: List[Dict[str, Any]]
        total_points = 0
        total_plane_points = 0
        total_corner_rays = 0
        stage_timing = {
            "request_build_ms": 0.0,
            "depth_batch_request_ms": 0.0,
            "plane_fit_ms": 0.0,
            "corner_ray_build_ms": 0.0,
            "ray_plane_intersection_ms": 0.0,
            "rectangle_reconstruct_ms": 0.0,
            "external_item_build_ms": 0.0,
        }  # type: Dict[str, float]
        bridge_debug = {
            "mode": "m41_2_roi_snapshot_intrinsics_rays",
            "point_count": 0,
            "plane_point_count": 0,
            "corner_ray_point_count": 0,
            "corner_ray_probe_count": 0,
            "corner_ray_mode": "intrinsics",
        }  # type: Dict[str, Any]

        for item_index, item in enumerate(items):
            try:
                request_started = time.perf_counter()
                plane_pixels = item.get("plane_sample_points") if isinstance(item.get("plane_sample_points"), list) else []
                if not plane_pixels:
                    raise GeometryError("no top-plane depth sample pixels")
                quad = item.get("quad") if isinstance(item.get("quad"), list) else []
                if len(quad) != 4:
                    raise GeometryError("top quadrilateral must contain four corners")

                request_points = []  # type: List[List[float]]
                for raw in plane_pixels:
                    u, v = self._point_xy(raw)
                    request_points.append([u, v, u, v])
                plane_count = len(request_points)
                stage_timing["request_build_ms"] += (time.perf_counter() - request_started) * 1000.0

                depth_started = time.perf_counter()
                plane_samples, response = self.bridge.sample_deproject(
                    request_points,
                    image_width,
                    image_height,
                    self.algorithm.depth_radius_px,
                    self.algorithm.depth_percentile,
                    self.algorithm.depth_min_valid_pixels,
                    self.algorithm.min_depth_mm,
                    self.algorithm.max_depth_mm,
                )
                stage_timing["depth_batch_request_ms"] += (time.perf_counter() - depth_started) * 1000.0
                if len(plane_samples) != plane_count:
                    raise UpstreamError("top-plane depth sample count mismatch")
                total_points += plane_count
                total_plane_points += plane_count

                plane_started = time.perf_counter()
                positions = [sample.get("position_camera") or [0.0, 0.0, 0.0] for sample in plane_samples]
                plane = self.algorithm.fit_plane(positions)
                stage_timing["plane_fit_ms"] += (time.perf_counter() - plane_started) * 1000.0

                ray_started = time.perf_counter()
                intrinsics = response.get("intrinsics") if isinstance(response.get("intrinsics"), Mapping) else None
                if intrinsics is None:
                    raise GeometryError("camera intrinsics unavailable for M41.2 corner rays")
                try:
                    depth_width = int(response.get("depth_width") or image_width)
                    depth_height = int(response.get("depth_height") or image_height)
                except (TypeError, ValueError, OverflowError):
                    depth_width, depth_height = image_width, image_height
                ray_points = self.algorithm.corner_rays_from_intrinsics(
                    quad,
                    intrinsics,
                    image_width,
                    image_height,
                    depth_width,
                    depth_height,
                    bool(response.get("flip_horizontal")),
                    bool(response.get("flip_vertical")),
                )
                total_corner_rays += len(ray_points)
                stage_timing["corner_ray_build_ms"] += (time.perf_counter() - ray_started) * 1000.0

                intersect_started = time.perf_counter()
                corners = self.algorithm.intersect_corner_rays(ray_points, plane)
                stage_timing["ray_plane_intersection_ms"] += (time.perf_counter() - intersect_started) * 1000.0

                rectangle_started = time.perf_counter()
                rectangle = self.algorithm.reconstruct_rectangle(quad, corners, plane)
                stage_timing["rectangle_reconstruct_ms"] += (time.perf_counter() - rectangle_started) * 1000.0

                build_started = time.perf_counter()
                external_items.append(
                    self.algorithm.build_external_item(item_index, item, plane, rectangle, plane_samples)
                )
                stage_timing["external_item_build_ms"] += (time.perf_counter() - build_started) * 1000.0

                client_timing = response.get("_client_timing") if isinstance(response.get("_client_timing"), Mapping) else {}
                bridge_debug.update({
                    "depth_age_ms": response.get("depth_age_ms"),
                    "depth_sequence": response.get("depth_sequence"),
                    "sample_ms": response.get("sample_ms"),
                    "snapshot_copy_ms": response.get("snapshot_copy_ms"),
                    "vectorized_sample_ms": response.get("vectorized_sample_ms"),
                    "vectorized_deproject_ms": response.get("vectorized_deproject_ms"),
                    "snapshot_attempts": response.get("snapshot_attempts"),
                    "snapshot_roi_px": response.get("snapshot_roi_px"),
                    "snapshot_roi_bytes": response.get("snapshot_roi_bytes"),
                    "roundtrip_ms": client_timing.get("roundtrip_ms"),
                    "connect_ms": client_timing.get("connect_ms"),
                    "send_ms": client_timing.get("send_ms"),
                    "transport": client_timing.get("transport") or response.get("mode"),
                    "headers_wait_ms": client_timing.get("headers_wait_ms"),
                    "body_read_ms": client_timing.get("body_read_ms"),
                    "json_decode_ms": client_timing.get("json_decode_ms"),
                    "response_bytes": client_timing.get("response_bytes"),
                    "combined_request_ok": bool(response.get("ok")),
                    "corner_ray_probe_count": 0,
                })
            except CameraUnavailableError:
                raise
            except Exception as error:
                ignored.append({
                    "id": str(item.get("source_id") or "candidate-{}".format(item_index)),
                    "reason": "m41_2_geometry_failed",
                    "message": str(error),
                })

        bridge_debug["point_count"] = total_points
        bridge_debug["plane_point_count"] = total_plane_points
        bridge_debug["corner_ray_point_count"] = total_corner_rays
        bridge_debug["corner_ray_probe_count"] = 0
        bridge_debug["stage_timing_ms"] = {
            key: round(float(value), 3) for key, value in stage_timing.items()
        }
        return external_items, ignored, bridge_debug

    def _camera_error_if_disconnected(self, error: Exception) -> Exception:
        if isinstance(error, CameraUnavailableError):
            return error
        try:
            health = self.bridge.health()
            connected = health.get("camera_connected") is not False and health.get("camera_started") is not False
            if not connected:
                return CameraUnavailableError("camera bridge reports camera disconnected")
        except CameraUnavailableError as camera_error:
            return camera_error
        except Exception:
            pass
        return error

    def _complete_packet(self, packet: InferencePacket) -> Dict[str, Any]:
        self.state.begin_postprocess()
        postprocess_started = time.perf_counter()
        timing = {
            # Compatibility: runtime_http_ms remains the complete client request
            # round trip, but it no longer includes JSON decoding.
            "runtime_http_ms": round(packet.runtime_http_ms, 3),
            "runtime_roundtrip_ms": round(packet.runtime_http_ms, 3),
            "runtime_lock_wait_ms": round(packet.runtime_lock_wait_ms, 3),
            "runtime_connect_ms": round(packet.runtime_connect_ms, 3),
            "runtime_send_ms": round(packet.runtime_send_ms, 3),
            "runtime_transport": packet.runtime_transport,
            "runtime_headers_wait_ms": round(packet.runtime_headers_wait_ms, 3),
            "runtime_body_read_ms": round(packet.runtime_body_read_ms, 3),
            "runtime_response_bytes": int(packet.runtime_response_bytes),
            "runtime_server_queue_ms": round(packet.runtime_server_queue_ms, 3),
            "runtime_server_route_ms": round(packet.runtime_server_route_ms, 3),
        }  # type: Dict[str, Any]
        try:
            if packet.error is not None:
                raise self._camera_error_if_disconnected(packet.error)
            runtime_result = packet.runtime_result
            if runtime_result is None:
                decode_started = time.perf_counter()
                runtime_result = self.runtime.decode_inference(packet.runtime_raw)
                packet.runtime_json_decode_ms = (time.perf_counter() - decode_started) * 1000.0
                packet.runtime_result = runtime_result
            packet.runtime_internal_ms = self._runtime_internal_ms(runtime_result)
            self._validate_runtime(runtime_result)
            timing["runtime_json_decode_ms"] = round(packet.runtime_json_decode_ms, 3)
            timing["runtime_internal_ms"] = round(packet.runtime_internal_ms, 3)
            timing["runtime_transport_overhead_ms"] = round(
                max(0.0, packet.runtime_http_ms - packet.runtime_internal_ms),
                3,
            )
            timing["runtime_non_route_ms"] = round(
                max(0.0, packet.runtime_http_ms - packet.runtime_server_queue_ms - packet.runtime_server_route_ms),
                3,
            )

            classify_started = time.perf_counter()
            classified = self.algorithm.classify(runtime_result)
            timing["classify_ms"] = round((time.perf_counter() - classify_started) * 1000.0, 3)
            selected_prepare = {
                "quad_fit_ms": 0.0,
                "interior_sample_ms": 0.0,
            }
            for classified_item in classified.items:
                prepare = classified_item.get("prepare_timing_ms") if isinstance(classified_item.get("prepare_timing_ms"), Mapping) else {}
                selected_prepare["quad_fit_ms"] += float(prepare.get("quad_fit_ms") or 0.0)
                selected_prepare["interior_sample_ms"] += float(prepare.get("interior_sample_ms") or 0.0)
            timing["quad_fit_ms"] = round(selected_prepare["quad_fit_ms"], 3)
            timing["interior_sample_ms"] = round(selected_prepare["interior_sample_ms"], 3)
            timing["classify_other_ms"] = round(
                max(0.0, timing["classify_ms"] - timing["quad_fit_ms"] - timing["interior_sample_ms"]),
                3,
            )

            geometry_started = time.perf_counter()
            external_items, geometry_ignored, bridge_debug = self._geometry_for_items(
                classified.items,
                classified.image_width,
                classified.image_height,
            )
            timing["geometry_3d_ms"] = round((time.perf_counter() - geometry_started) * 1000.0, 3)
            if isinstance(bridge_debug, Mapping):
                stage = bridge_debug.get("stage_timing_ms") if isinstance(bridge_debug.get("stage_timing_ms"), Mapping) else {}
                timing["geometry_request_build_ms"] = round(float(stage.get("request_build_ms") or 0.0), 3)
                timing["depth_sample_deproject_ms"] = round(float(stage.get("depth_batch_request_ms") or 0.0), 3)
                timing["plane_fit_ms"] = round(float(stage.get("plane_fit_ms") or 0.0), 3)
                timing["corner_ray_build_ms"] = round(float(stage.get("corner_ray_build_ms") or 0.0), 3)
                timing["ray_plane_intersection_ms"] = round(float(stage.get("ray_plane_intersection_ms") or 0.0), 3)
                timing["rectangle_reconstruct_ms"] = round(float(stage.get("rectangle_reconstruct_ms") or 0.0), 3)
                timing["external_item_build_ms"] = round(float(stage.get("external_item_build_ms") or 0.0), 3)
                accounted_geometry = sum(
                    float(timing.get(key) or 0.0)
                    for key in (
                        "geometry_request_build_ms",
                        "depth_sample_deproject_ms",
                        "plane_fit_ms",
                        "corner_ray_build_ms",
                        "ray_plane_intersection_ms",
                        "rectangle_reconstruct_ms",
                        "external_item_build_ms",
                    )
                )
                timing["geometry_other_ms"] = round(max(0.0, timing["geometry_3d_ms"] - accounted_geometry), 3)
                timing["depth_bridge_internal_ms"] = round(float(bridge_debug.get("sample_ms") or 0.0), 3)
                timing["depth_snapshot_copy_ms"] = round(float(bridge_debug.get("snapshot_copy_ms") or 0.0), 3)
                timing["depth_vectorized_sample_ms"] = round(float(bridge_debug.get("vectorized_sample_ms") or 0.0), 3)
                timing["depth_vectorized_deproject_ms"] = round(float(bridge_debug.get("vectorized_deproject_ms") or 0.0), 3)
                timing["depth_snapshot_attempts"] = int(bridge_debug.get("snapshot_attempts") or 0)
                timing["depth_snapshot_roi_px"] = list(bridge_debug.get("snapshot_roi_px") or [])
                timing["depth_snapshot_roi_bytes"] = int(bridge_debug.get("snapshot_roi_bytes") or 0)
                timing["depth_transport"] = str(bridge_debug.get("transport") or bridge_debug.get("mode") or "unknown")
                timing["depth_connect_ms"] = round(float(bridge_debug.get("connect_ms") or 0.0), 3)
                timing["depth_send_ms"] = round(float(bridge_debug.get("send_ms") or 0.0), 3)
                timing["depth_http_roundtrip_ms"] = round(float(bridge_debug.get("roundtrip_ms") or 0.0), 3)
                timing["depth_http_headers_wait_ms"] = round(float(bridge_debug.get("headers_wait_ms") or 0.0), 3)
                timing["depth_http_body_read_ms"] = round(float(bridge_debug.get("body_read_ms") or 0.0), 3)
                timing["depth_json_decode_ms"] = round(float(bridge_debug.get("json_decode_ms") or 0.0), 3)
                timing["depth_response_bytes"] = int(bridge_debug.get("response_bytes") or 0)
                timing["depth_point_count"] = int(bridge_debug.get("point_count") or 0)
                timing["depth_plane_point_count"] = int(bridge_debug.get("plane_point_count") or 0)
                timing["corner_ray_point_count"] = int(bridge_debug.get("corner_ray_point_count") or 0)
                timing["corner_ray_probe_count"] = int(bridge_debug.get("corner_ray_probe_count") or 0)
                timing["corner_ray_mode"] = str(bridge_debug.get("corner_ray_mode") or "unknown")

            build_started = time.perf_counter()

            try:
                capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
            except (TypeError, ValueError, OverflowError):
                capture_timestamp_ms = 0
            timestamp = capture_timestamp_ms / 1000.0 if capture_timestamp_ms > 0 else packet.started_at
            protocol_items = [
                grasp_point
                for item in external_items
                for grasp_point in self._build_grasp_point_items(item)
            ]
            robot = {
                "type": "detection",
                "frame_id": packet.frame_id,
                "timestamp": timestamp,
                "items": protocol_items,
                "fault_code": FAULT_NONE,
                "fault_type": FAULT_TYPE_NONE,
            }
            if packet.request_id is not None:
                robot["request_id"] = packet.request_id

            # Runtime segmentation remains untouched; M41.2 geometry is appended.
            visualization = dict(runtime_result)
            visualization["carton_bundle_grasp"] = {
                "version": "M41.2",
                "geometry_mode": "FULL_TOP_FIXED_SIZE",
                "items": external_items,
                "ignored": list(classified.ignored) + list(geometry_ignored),
                "bundle_prior_mm": [self.algorithm.length_mm, self.algorithm.width_mm],
            }
            producer = self.producer_metadata()
            visualization["producer"] = producer
            decision = {
                "schema_version": "1.0",
                "message_type": "app_decision",
                "status": "ok",
                "app_id": "carton_bundle_grasp_vision",
                "task": "segmentation_carton_bundle_grasp",
                "device_id": self.settings["device_id"],
                "component": self.settings["component"],
                "timestamp_ms": _timestamp_ms(),
                "frame_id": runtime_result.get("frame_id"),
                "result_id": runtime_result.get("result_id"),
                "robot_message": robot,
                "visualization_result": visualization,
                "producer": producer,
            }
            timing["result_build_ms"] = round((time.perf_counter() - build_started) * 1000.0, 3)
            timing["postprocess_stage_ms"] = round((time.perf_counter() - postprocess_started) * 1000.0, 3)
            timing["pipeline_age_ms"] = round((time.monotonic() - packet.started_monotonic) * 1000.0, 3)
            timing["total_ms"] = timing["pipeline_age_ms"]
            decision["app_timing"] = timing
            visualization["carton_bundle_grasp"]["app_timing"] = timing

            latency_ms = float(timing["total_ms"])
            self.state.success(decision, robot, runtime_result, latency_ms, timing)
            self._save_debug_async({
                "decision": decision,
                "runtime_result": runtime_result,
                "ignored": list(classified.ignored) + list(geometry_ignored),
                "bridge": bridge_debug,
            }, b"")
            return decision
        except Exception as raw_error:
            error = self._camera_error_if_disconnected(raw_error)
            latency_ms = (time.monotonic() - packet.started_monotonic) * 1000.0
            timing["postprocess_stage_ms"] = round((time.perf_counter() - postprocess_started) * 1000.0, 3)
            timing["total_ms"] = round(latency_ms, 3)
            decision, robot = self._error_result(packet.frame_id, packet.request_id, error, packet.started_at)
            decision["app_timing"] = timing
            self.state.failure(decision, robot, error, latency_ms)
            self._save_debug_async({"decision": decision, "error": str(error), "app_timing": timing}, b"")
            return decision
        finally:
            self.state.end_postprocess()

    def evaluate_once(self, request_id: object = None) -> Dict[str, Any]:
        # Manual/API triggers remain synchronous, but they only serialize against
        # other manual requests.  Runtime access itself is protected separately,
        # so the production pipeline remains correct with one RKNN context.
        with self.execution_lock:
            packet = self._run_inference_stage(request_id=request_id, continuous=False)
            return self._complete_packet(packet)

    def _dispatch_packet(self, packet: InferencePacket, decision: Mapping[str, Any]) -> None:
        robot = decision.get("robot_message") if isinstance(decision.get("robot_message"), Mapping) else {}
        if packet.trigger is not None:
            try:
                packet.trigger.session.send_json(robot)
            except OSError:
                pass
        elif packet.continuous and self.websocket.client_count() > 0:
            self.websocket.broadcast_json(robot)

    def _enqueue_packet(self, packet: InferencePacket) -> None:
        if packet.trigger is not None:
            try:
                self.result_queue.put(packet, timeout=max(0.1, float(self.settings["app"]["request_timeout_ms"]) / 1000.0))
            except queue.Full:
                self.state.counters["pipeline_trigger_drop"] += 1
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
            return
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
                    trigger=trigger,
                    continuous=trigger is None,
                )
                if self.pipeline_enabled:
                    self._enqueue_packet(packet)
                else:
                    decision = self._complete_packet(packet)
                    self._dispatch_packet(packet, decision)
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
            decision = self._complete_packet(packet)
            self._dispatch_packet(packet, decision)

    def _refresh_upstream_status(self) -> None:
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
        with self.status_cache_lock:
            self.cached_model_name = model_name
            self.cached_camera_connected = camera_connected
            self.cached_upstream_status_at = time.monotonic()

    def _status_message(self, refresh: bool = False) -> Dict[str, Any]:
        if refresh:
            self._refresh_upstream_status()
        snapshot = self.state.snapshot(self.websocket)
        with self.status_cache_lock:
            model_name = self.cached_model_name
            camera_connected = self.cached_camera_connected
            status_age_ms = (
                max(0.0, (time.monotonic() - self.cached_upstream_status_at) * 1000.0)
                if self.cached_upstream_status_at > 0
                else -1.0
            )
        fault_code, fault_type = self._external_fault(camera_connected)
        return {
            "type": "status",
            "task": "carton_bundle_grasp_vision",
            "online": True,
            "fps": snapshot["detection_fps"],
            "configured_fps": snapshot["configured_detection_fps"],
            "push_mode": "every_completed_result",
            "model": model_name,
            "camera_connected": camera_connected,
            "fault_code": fault_code,
            "fault_type": fault_type,
            "latency_ms": snapshot["last_latency_ms"],
            "continuous_enabled": snapshot["continuous_enabled"],
            "clients": snapshot["websocket"]["clients"],
            "video_url": snapshot["video"]["url"],
            "upstream_status_age_ms": round(status_age_ms, 3),
        }

    def _status_loop(self) -> None:
        interval = max(0.5, float(self.settings["websocket"].get("status_interval_s", 2.0)))
        self._refresh_upstream_status()
        while not self.stop_event.wait(interval):
            self._refresh_upstream_status()
            if self.websocket.client_count() > 0:
                self.websocket.broadcast_json(self._status_message())

    def _save_debug_async(self, document: Mapping[str, Any], depth_bytes: bytes) -> None:
        if not self.debug_enabled:
            return
        threading.Thread(target=self._save_debug, args=(deepcopy(dict(document)), bytes(depth_bytes)), name="carton-bundle-debug", daemon=True).start()

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
                    bundle_grasp = visualization.get("carton_bundle_grasp") if isinstance(visualization.get("carton_bundle_grasp"), Mapping) else {}
                    self._draw_overlay(rgb, bundle_grasp.get("items"), self.debug_root / "overlay.jpg")
            except Exception:
                pass

    @staticmethod
    def _draw_overlay(rgb: bytes, items_value: object, output: Path) -> None:
        image = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        items = items_value if isinstance(items_value, list) else []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            contour = np.asarray(item.get("contour_px") or [], dtype=np.int32)
            if contour.ndim == 2 and contour.shape[0] >= 3:
                cv2.polylines(image, [contour], True, (255, 255, 0), 1)

            observed = item.get("observed_corners_px") if isinstance(item.get("observed_corners_px"), Mapping) else {}
            regularized = item.get("regularized_corners_px") if isinstance(item.get("regularized_corners_px"), Mapping) else {}
            names = ("top_left", "top_right", "bottom_right", "bottom_left")
            obs_quad = np.asarray([observed.get(name) for name in names], dtype=np.float32)
            reg_quad = np.asarray([regularized.get(name) for name in names], dtype=np.float32)
            if obs_quad.shape == (4, 2):
                cv2.polylines(image, [np.rint(obs_quad).astype(np.int32)], True, (0, 165, 255), 2)
            if reg_quad.shape == (4, 2):
                cv2.polylines(image, [np.rint(reg_quad).astype(np.int32)], True, (0, 255, 0), 3)

            grasp = item.get("grasp_points_px") if isinstance(item.get("grasp_points_px"), Mapping) else {}
            for role, point in grasp.items():
                if isinstance(point, list) and len(point) >= 2:
                    x, y = int(round(float(point[0]))), int(round(float(point[1])))
                    cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(image, str(role), (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            observed_size = item.get("observed_size") if isinstance(item.get("observed_size"), Mapping) else {}
            top_plane = item.get("top_plane") if isinstance(item.get("top_plane"), Mapping) else {}
            label = "M41.2 L={:.1f} W={:.1f} RMS={:.2f}".format(
                float(observed_size.get("length_mm") or 0.0),
                float(observed_size.get("width_mm") or 0.0),
                float(top_plane.get("rms_mm") or 0.0),
            )
            cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(str(output), image)



class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsCartonBundleGrasp/1.1"

    @property
    def service(self) -> CartonBundleGraspVisionService:
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
            self._send(200, {"schema_version": "1.0", "message_type": "app_health", "status": "ok", "health": snapshot["health"], "app_id": "carton_bundle_grasp_vision", "timestamp_ms": _timestamp_ms()})
        elif path in {"/api/app/status", "/api/gateway/status", "/api/ws/status"}:
            snapshot["external_status"] = self.service._status_message()
            snapshot["pipeline"] = self.service.pipeline_status()
            snapshot["ipc"] = self.service.ipc_status()
            self._send(200, snapshot)
        elif path == "/api/ws/clients":
            self._send(200, {"status": "ok", "clients": self.service.websocket.client_snapshot()})
        elif path in {"/api/app/registers", "/api/gateway/registers"}:
            self._send(200, {"schema_version": "1.0", "message_type": "register_snapshot", "status": "not_applicable", "protocol": "websocket", "registers": []})
        elif path == "/api/app/latest_decision":
            latest = snapshot.get("latest_decision")
            if isinstance(latest, Mapping):
                latest = deepcopy(dict(latest))
                producer = self.service.producer_metadata()
                latest["producer"] = producer
                visualization = latest.get("visualization_result")
                if isinstance(visualization, dict):
                    visualization["producer"] = producer
            self._send(200, latest or {"status": "empty", "message_type": "app_decision"})
        elif path == "/api/app/inference_settings":
            self._send(200, self.service.inference_settings())
        elif path == "/api/app/latest_gateway_message":
            self._send(200, snapshot.get("latest_gateway_message") or {"status": "empty", "type": "detection", "items": []})
        else:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/api/app/evaluate_once", "/api/task/evaluate_once", "/api/app/inference_settings"}:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})
            return
        try:
            document = self._read_json()
            if path == "/api/app/inference_settings":
                self._send(200, self.service.set_production_fps(document.get("detection_fps")))
                return
            request_id = document.get("request_id")
            if request_id is None:
                self.service.manual_request_id += 1
                request_id = "manual-{}".format(self.service.manual_request_id)
            self._send(200, self.service.evaluate_once(request_id))
        except ValueError as error:
            self._send(400, {"status": "error", "error": {"code": "INVALID_INFERENCE_SETTINGS", "message": str(error)}})
        except Exception as error:
            self._send(500, {"status": "error", "error": {"code": type(error).__name__, "message": str(error)}})


def run(config: Mapping[str, Any]) -> int:
    service = CartonBundleGraspVisionService(config)
    http = config["carton_bundle_grasp"]["app"]
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
    thread = threading.Thread(target=server.serve_forever, name="carton-bundle-http", daemon=True)
    thread.start()
    ws = config["carton_bundle_grasp"]["websocket"]
    print(
        "Carton Bundle Grasp M41.2 started: ws={}:{}{} http={}:{} runtime={} video={}".format(
            ws["listen_host"], ws["listen_port"], ws["path"], http["listen_host"], http["listen_port"],
            config["carton_bundle_grasp"]["runtime"]["url"], config["carton_bundle_grasp"]["video"]["public_url"]
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
    parser = argparse.ArgumentParser(description="M41.2 carton bundle top-plane grasp WebSocket service")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
