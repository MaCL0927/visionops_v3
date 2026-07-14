#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External-box WebSocket service for the tube_pick_vision task."""
from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.config import PROJECT_ROOT, load_config
from production.carton_line.gateway.runtime_client import HttpClient, RuntimeClient, UpstreamError
from production.carton_line.tasks.tube_pick_vision.algorithm import TubePickAlgorithm
from production.carton_line.tasks.tube_pick_vision.depth_coordinate import BridgeCoordinateClient
from production.carton_line.tasks.tube_pick_vision.websocket_server import WebSocketJsonServer, WebSocketSession


DEFAULT_CONFIG = PROJECT_ROOT / "production/carton_line/config/line.yaml"
MAX_HTTP_BODY = 1024 * 1024


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
    request_id: object


class ServiceState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.continuous_enabled = bool(config["pick"]["websocket"].get("auto_start", True))
        self.busy = False
        self.frame_id = 0
        self.latest_detection: dict[str, Any] | None = None
        self.latest_runtime_result: dict[str, Any] | None = None
        self.latest_debug: dict[str, Any] | None = None
        self.last_error: dict[str, Any] | None = None
        self.last_latency_ms = 0.0
        self.counters: dict[str, int] = defaultdict(int)
        self.inference_times: deque[float] = deque(maxlen=100)

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

    def success(
        self,
        detection: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        debug: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        with self.lock:
            self.busy = False
            self.latest_detection = dict(detection)
            self.latest_runtime_result = dict(runtime_result)
            self.latest_debug = dict(debug)
            self.last_error = None
            self.last_latency_ms = float(latency_ms)
            self.inference_times.append(time.monotonic())
            self.counters["inference_success"] += 1

    def failure(self, detection: Mapping[str, Any], error: Exception, latency_ms: float) -> None:
        with self.lock:
            self.busy = False
            self.latest_detection = dict(detection)
            self.last_latency_ms = float(latency_ms)
            self.last_error = {
                "code": type(error).__name__,
                "message": str(error),
                "timestamp_ms": _timestamp_ms(),
            }
            self.counters["inference_failure"] += 1

    def fps(self) -> float:
        with self.lock:
            times = list(self.inference_times)
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return round((len(times) - 1) / elapsed, 3) if elapsed > 0 else 0.0

    def snapshot(self, websocket: WebSocketJsonServer | None = None) -> dict[str, Any]:
        ws_config = self.config["pick"]["websocket"]
        with self.lock:
            health = "degraded" if self.last_error else "ok"
            snapshot = {
                "schema_version": "1.0",
                "message_type": "tube_pick_service_status",
                "status": "ok",
                "health": health,
                "timestamp_ms": _timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "continuous_enabled": self.continuous_enabled,
                "detection_fps": self.fps(),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "websocket": {
                    "listen_host": ws_config["listen_host"],
                    "listen_port": ws_config["listen_port"],
                    "path": ws_config["path"],
                    "clients": websocket.client_count() if websocket else 0,
                },
                "video": {
                    "type": "mjpeg",
                    "url": self.config["pick"]["video"]["public_url"],
                    "sync": "soft",
                },
                "runtime_url": self.config["runtimes"]["pick"]["url"],
                "latest_detection": self.latest_detection,
                "last_error": self.last_error,
                "counters": dict(self.counters),
            }
        return snapshot


class TubePickVisionService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.settings = config["pick"]
        timeout_s = int(config["service"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeClient(config["runtimes"]["pick"]["url"], timeout_s)
        self.http = HttpClient(timeout_s=timeout_s)
        self.algorithm = TubePickAlgorithm(self.settings["algorithm"])
        bridge_base = str(config["camera_bridge"]["base_url"]).rstrip("/")
        self.bridge = BridgeCoordinateClient(
            self.http,
            str(config["camera_bridge"]["depth_url"]),
            bridge_base + str(config["camera_bridge"]["health_path"]),
            str(config["camera_bridge"]["deproject_url"]),
            self.algorithm.max_age_ms,
        )
        self.state = ServiceState(config)
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.execution_lock = threading.Lock()
        self.debug_lock = threading.Lock()
        self.trigger_queue: "queue.Queue[TriggerRequest]" = queue.Queue(
            maxsize=int(self.settings["websocket"].get("trigger_queue_size", 32))
        )
        self.worker_thread: threading.Thread | None = None
        self.status_thread: threading.Thread | None = None
        self.manual_request_id = 0
        debug = self.settings.get("debug") if isinstance(self.settings.get("debug"), Mapping) else {}
        self.debug_enabled = bool(debug.get("save_every_trigger", True))
        self.debug_root = Path(str(debug.get("save_root", "/tmp/visionops_v3/carton_line/tube_pick_vision/latest")))

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
        self.worker_thread = threading.Thread(target=self._worker_loop, name="tube-pick-inference", daemon=True)
        self.status_thread = threading.Thread(target=self._status_loop, name="tube-pick-status", daemon=True)
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

    @staticmethod
    def _valid_request_id(value: object) -> bool:
        return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) != ""

    def _ack(
        self,
        session: WebSocketSession,
        request_type: str,
        success: bool,
        request_id: object | None = None,
        **extra: Any,
    ) -> None:
        document: dict[str, Any] = {
            "type": "ack",
            "request_type": request_type,
            "success": bool(success),
            "timestamp": time.time(),
        }
        if request_id is not None:
            document["request_id"] = request_id
        document.update(extra)
        session.send_json(document)

    def _on_ws_json(self, session: WebSocketSession, document: dict[str, Any]) -> None:
        message_type = str(document.get("type") or "")
        if message_type == "control":
            command = str(document.get("command") or "").lower()
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
                    self._ack(
                        session,
                        "control",
                        False,
                        request_id,
                        command=command,
                        error="trigger 必须携带非空 request_id",
                    )
                    return
                try:
                    self.trigger_queue.put_nowait(TriggerRequest(session=session, request_id=request_id))
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
        if message_type == "config":
            self._ack(
                session,
                "config",
                False,
                document.get("request_id"),
                error="ROI/threshold 仅由 VisionOps Web 与模型配置管理",
            )
            return
        self._ack(session, message_type or "unknown", False, document.get("request_id"), error="unsupported message type")

    def _validate_runtime(self, result: Mapping[str, Any]) -> None:
        runtime = self.config["runtimes"]["pick"]
        task_type = str(result.get("task_type") or "").lower()
        accepted_types = set(runtime.get("accepted_task_types", []))
        if accepted_types and task_type not in accepted_types:
            raise ValueError(f"pick Runtime task_type={task_type!r} 不在白名单 {sorted(accepted_types)}")
        accepted_models = set(runtime.get("accepted_model_ids", [])) | set(runtime.get("accepted_model_names", []))
        if accepted_models and not (accepted_models & _path_models(result)):
            raise ValueError(
                f"pick Runtime 当前模型不在白名单: current={sorted(_path_models(result))}, "
                f"accepted={sorted(accepted_models)}"
            )

    def _new_error_detection(self, frame_id: int, request_id: object | None, error: Exception, started_at: float) -> dict[str, Any]:
        detection: dict[str, Any] = {
            "type": "detection",
            "frame_id": frame_id,
            "timestamp": started_at,
            "items": [],
            "latency_ms": round((time.time() - started_at) * 1000.0, 3),
            "error": {"code": type(error).__name__, "message": str(error)},
        }
        if request_id is not None:
            detection["request_id"] = request_id
        return detection

    def evaluate_once(self, request_id: object | None = None) -> dict[str, Any]:
        frame_id = self.state.next_frame_id()
        started_at = time.time()
        self.state.begin()
        started_monotonic = time.monotonic()
        with self.execution_lock:
            try:
                runtime_result = self.runtime.infer_once()
                self._validate_runtime(runtime_result)
                classified = self.algorithm.classify(runtime_result)
                sampled: list[dict[str, Any]] = []
                positions: list[list[float]] = []
                bridge_debug: dict[str, Any] = {}
                depth_bytes = b""
                if classified.items:
                    depth, bridge_health, depth_bytes = self.bridge.get_depth()
                    sampled = self.algorithm.sample_items(classified, depth)
                    deproject_input = [
                        [float(item["center_x"]), float(item["center_y"]), float(item["z_mm"])]
                        for item in sampled
                    ]
                    positions, deproject_result = self.bridge.deproject(deproject_input)
                    bridge_debug = {
                        "health": bridge_health,
                        "deproject": deproject_result,
                    }
                items = self.algorithm.build_external_items(sampled, positions) if sampled else []
                latency_ms = (time.monotonic() - started_monotonic) * 1000.0
                try:
                    capture_timestamp_ms = int(runtime_result.get("capture_timestamp_ms") or 0)
                except (TypeError, ValueError, OverflowError):
                    capture_timestamp_ms = 0
                capture_timestamp = (capture_timestamp_ms / 1000.0) if capture_timestamp_ms > 0 else started_at
                detection: dict[str, Any] = {
                    "type": "detection",
                    "frame_id": frame_id,
                    "timestamp": capture_timestamp,
                    "items": items,
                    "image": {
                        "width": classified.image_width,
                        "height": classified.image_height,
                    },
                    "coordinate_frame": "color_camera",
                    "coordinate_unit": "mm",
                    "video_url": self.settings["video"]["public_url"],
                    "video_sync": "soft",
                    "latency_ms": round(latency_ms, 3),
                    "source": {
                        "runtime_frame_id": runtime_result.get("frame_id"),
                        "runtime_result_id": runtime_result.get("result_id"),
                    },
                }
                if request_id is not None:
                    detection["request_id"] = request_id
                debug = {
                    "detection": detection,
                    "runtime_result": runtime_result,
                    "sampled_items": sampled,
                    "ignored_detections": classified.ignored,
                    "bridge": bridge_debug,
                }
                self.state.success(detection, runtime_result, debug, latency_ms)
                self._save_debug_async(debug, depth_bytes)
                return detection
            except Exception as error:
                latency_ms = (time.monotonic() - started_monotonic) * 1000.0
                detection = self._new_error_detection(frame_id, request_id, error, started_at)
                self.state.failure(detection, error, latency_ms)
                self._save_debug_async({"detection": detection, "error": str(error)}, b"")
                return detection

    def _worker_loop(self) -> None:
        frequency_hz = float(self.settings["websocket"].get("detection_hz", 10.0))
        period_s = 1.0 / max(0.1, frequency_hz)
        next_continuous = time.monotonic()
        while not self.stop_event.is_set():
            try:
                trigger = self.trigger_queue.get_nowait()
            except queue.Empty:
                trigger = None
            if trigger is not None:
                detection = self.evaluate_once(trigger.request_id)
                try:
                    trigger.session.send_json(detection)
                except OSError:
                    pass
                continue

            continuous = self.state.continuous_enabled and self.websocket.client_count() > 0
            now = time.monotonic()
            if continuous and now >= next_continuous:
                detection = self.evaluate_once(None)
                self.websocket.broadcast_json(detection)
                next_continuous = max(next_continuous + period_s, time.monotonic())
                continue
            timeout = 0.2
            if continuous:
                timeout = max(0.01, min(0.2, next_continuous - now))
            self.wakeup.wait(timeout)
            self.wakeup.clear()
            if not continuous:
                next_continuous = time.monotonic()

    def _status_message(self) -> dict[str, Any]:
        snapshot = self.state.snapshot(self.websocket)
        model_name = ""
        camera_connected = False
        try:
            runtime = self.runtime.status()
            loaded_model = runtime.get("loaded_model") if isinstance(runtime.get("loaded_model"), Mapping) else {}
            model_name = str(loaded_model.get("model_name") or loaded_model.get("model_id") or "")
            camera_connected = bool(runtime.get("camera_connected"))
        except Exception:
            pass
        return {
            "type": "status",
            "online": True,
            "fps": snapshot["detection_fps"],
            "model": model_name,
            "camera_connected": camera_connected,
            "latency_ms": snapshot["last_latency_ms"],
            "continuous_enabled": snapshot["continuous_enabled"],
            "clients": snapshot["websocket"]["clients"],
            "video_url": snapshot["video"]["url"],
            "error": snapshot["last_error"],
        }

    def _status_loop(self) -> None:
        interval = max(0.5, float(self.settings["websocket"].get("status_interval_s", 2.0)))
        while not self.stop_event.wait(interval):
            if self.websocket.client_count() > 0:
                self.websocket.broadcast_json(self._status_message())

    def _save_debug_async(self, document: Mapping[str, Any], depth_bytes: bytes) -> None:
        if not self.debug_enabled:
            return
        snapshot = dict(document)
        threading.Thread(
            target=self._save_debug,
            args=(snapshot, bytes(depth_bytes)),
            name="tube-pick-debug-writer",
            daemon=True,
        ).start()

    def _save_debug(self, document: Mapping[str, Any], depth_bytes: bytes) -> None:
        with self.debug_lock:
            self.debug_root.mkdir(parents=True, exist_ok=True)
            (self.debug_root / "result.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if depth_bytes:
                (self.debug_root / "depth.png").write_bytes(depth_bytes)
            try:
                runtime_result = document.get("runtime_result") if isinstance(document.get("runtime_result"), Mapping) else {}
                rgb = self.runtime.snapshot()
                if rgb:
                    (self.debug_root / "rgb.jpg").write_bytes(rgb)
                    self._draw_overlay(rgb, runtime_result, document.get("sampled_items"), self.debug_root / "overlay.jpg")
            except Exception:
                pass

    @staticmethod
    def _draw_overlay(rgb_bytes: bytes, runtime_result: Mapping[str, Any], sampled_items: object, output_path: Path) -> None:
        image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        sampled = sampled_items if isinstance(sampled_items, list) else []
        sample_by_source = {
            str(item.get("source_id")): item for item in sampled if isinstance(item, Mapping)
        }
        detections = runtime_result.get("detections") if isinstance(runtime_result.get("detections"), list) else []
        for index, raw in enumerate(detections):
            if not isinstance(raw, Mapping):
                continue
            bbox = raw.get("bbox_xyxy")
            center = raw.get("center_xy")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            class_id = int(raw.get("class_id") or 0)
            score = float(raw.get("score") or 0.0)
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            visual_by_class = {
                0: ("product", (0, 255, 0)),
                1: ("separator", (255, 180, 0)),
                2: ("lying", (0, 0, 255)),
            }
            label, color = visual_by_class.get(class_id, (f"class-{class_id}", (200, 200, 200)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            z_mm = 0
            source_id = str(raw.get("id") or f"det-{index}")
            sample = sample_by_source.get(source_id, {})
            if isinstance(sample, Mapping):
                z_mm = int(sample.get("z_mm") or 0)
            cv2.putText(
                image,
                f"{label} {score:.2f} z={z_mm}mm",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            if isinstance(center, list) and len(center) >= 2:
                cv2.circle(image, (int(round(float(center[0]))), int(round(float(center[1])))), 4, (0, 0, 255), -1)
        cv2.imwrite(str(output_path), image)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsTubePickWS/2.0"

    @property
    def service(self) -> TubePickVisionService:
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

    def _read_json(self) -> dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length 非法") from error
        if size < 0 or size > MAX_HTTP_BODY:
            raise ValueError("请求体超过限制")
        if size == 0:
            return {}
        try:
            document = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("请求体必须是 JSON 对象") from error
        if not isinstance(document, dict):
            raise ValueError("请求体顶层必须是对象")
        return document

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            snapshot = self.service.state.snapshot(self.service.websocket)
            self._send(200, {"ok": True, "status": snapshot["health"], "component": "tube_pick_vision_ws"})
        elif path in {"/api/app/status", "/api/gateway/status", "/api/ws/status"}:
            self._send(200, self.service.state.snapshot(self.service.websocket))
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
        elif path in {"/api/app/latest_decision", "/api/app/latest_gateway_message"}:
            response = self.service.state.snapshot(self.service.websocket).get("latest_detection")
            self._send(200, response or {"status": "empty", "type": "detection", "items": []})
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
                request_id = f"manual-{self.service.manual_request_id}"
            response = self.service.evaluate_once(request_id)
            self._send(200, response)
        except Exception as error:
            self._send(500, {"status": "error", "error": {"code": type(error).__name__, "message": str(error)}})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps tube-pick external-box WebSocket server")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="carton_line unified YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    service = TubePickVisionService(config)
    http_config = config["pick"]["http"]
    server = ReusableThreadingHTTPServer(
        (str(http_config["listen_host"]), int(http_config["listen_port"])), StatusHandler
    )
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
    http_thread = threading.Thread(target=server.serve_forever, name="tube-pick-http", daemon=True)
    http_thread.start()
    ws = config["pick"]["websocket"]
    print(
        "Tube Pick Vision WebSocket started: "
        f"ws={ws['listen_host']}:{ws['listen_port']}{ws['path']} "
        f"http={http_config['listen_host']}:{http_config['listen_port']} "
        f"runtime={config['runtimes']['pick']['url']} "
        f"video={config['pick']['video']['public_url']}"
    )
    try:
        while not stop_once.wait(1.0):
            pass
    finally:
        service.stop()
        server.shutdown()
        server.server_close()
        http_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
