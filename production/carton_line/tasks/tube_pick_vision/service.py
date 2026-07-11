#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCP-triggered tube product / large-separator production service."""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import cv2  # type: ignore
import numpy as np  # type: ignore

from production.carton_line.gateway.config import PROJECT_ROOT, load_config
from production.carton_line.gateway.runtime_client import HttpClient, RuntimeClient, UpstreamError
from production.carton_line.tasks.tube_pick_vision.algorithm import TubePickAlgorithm, decode_depth_png
from production.carton_line.tasks.tube_pick_vision.tcp_client import (
    ReconnectingJsonTcpClient,
    StarHashJsonCodec,
)


DEFAULT_CONFIG = PROJECT_ROOT / "production/carton_line/config/line.yaml"
MAX_HTTP_BODY = 1024 * 1024


def _timestamp_pair() -> list[int]:
    now_ns = time.time_ns()
    return [int(now_ns // 1_000_000_000), int(now_ns % 1_000_000_000)]


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default


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


class ServiceState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.connection_state = "starting"
        self.connection_detail: dict[str, Any] = {}
        self.busy = False
        self.latest_request: dict[str, Any] | None = None
        self.latest_response: dict[str, Any] | None = None
        self.latest_debug: dict[str, Any] | None = None
        self.last_error: dict[str, Any] | None = None
        self.counters: dict[str, int] = defaultdict(int)

    def connection(self, state: str, detail: Mapping[str, Any]) -> None:
        with self.lock:
            self.connection_state = state
            self.connection_detail = dict(detail)
            if state == "connected":
                self.counters["connections"] += 1
            elif state in {"disconnected", "client_error"}:
                self.counters["disconnects"] += 1

    def begin(self, request: Mapping[str, Any]) -> None:
        with self.lock:
            self.busy = True
            self.latest_request = dict(request)
            self.counters["triggers"] += 1

    def success(self, response: Mapping[str, Any], debug: Mapping[str, Any]) -> None:
        with self.lock:
            self.busy = False
            self.latest_response = dict(response)
            self.latest_debug = dict(debug)
            self.last_error = None
            self.counters["success"] += 1

    def failure(self, response: Mapping[str, Any], error: Exception) -> None:
        with self.lock:
            self.busy = False
            self.latest_response = dict(response)
            self.last_error = {
                "code": type(error).__name__,
                "message": str(error),
                "timestamp_ms": _timestamp_ms(),
            }
            self.counters["failure"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            tcp = self.config["pick"]["tcp"]
            return {
                "schema_version": "1.0",
                "message_type": "tube_pick_service_status",
                "status": "ok",
                "health": "degraded" if (self.last_error or self.connection_state != "connected") else "ok",
                "timestamp_ms": _timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "busy": self.busy,
                "tcp_client": {
                    "state": self.connection_state,
                    "detail": dict(self.connection_detail),
                    "server_host": tcp["server_host"],
                    "server_port": tcp["server_port"],
                },
                "runtime_url": self.config["runtimes"]["pick"]["url"],
                "latest_request": self.latest_request,
                "latest_response": self.latest_response,
                "last_error": self.last_error,
                "counters": dict(self.counters),
            }


class TubePickVisionService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.settings = config["pick"]
        timeout_s = int(config["service"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeClient(config["runtimes"]["pick"]["url"], timeout_s)
        self.http = HttpClient(timeout_s=timeout_s)
        self.algorithm = TubePickAlgorithm(self.settings["algorithm"])
        self.depth_url = str(config["camera_bridge"]["depth_url"])
        self.depth_meta_url = str(config["camera_bridge"]["depth_meta_url"])
        self.state = ServiceState(config)
        self.stop_event = threading.Event()
        self.cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.cache_lock = threading.RLock()
        self.execution_lock = threading.Lock()
        self.debug_lock = threading.Lock()
        self.manual_trigger_index = 0
        debug = self.settings.get("debug") if isinstance(self.settings.get("debug"), Mapping) else {}
        self.debug_enabled = bool(debug.get("save_every_trigger", True))
        self.debug_root = Path(str(debug.get("save_root", "/tmp/visionops_v3/carton_line/tube_pick_vision/latest")))

        tcp = self.settings["tcp"]
        self.client = ReconnectingJsonTcpClient(
            host=str(tcp["server_host"]),
            port=int(tcp["server_port"]),
            on_message=self.handle_message,
            on_state=self.state.connection,
            connect_timeout_s=int(tcp["connect_timeout_ms"]) / 1000.0,
            read_timeout_s=int(tcp["read_timeout_ms"]) / 1000.0,
            reconnect_initial_s=int(tcp["reconnect_initial_ms"]) / 1000.0,
            reconnect_max_s=int(tcp["reconnect_max_ms"]) / 1000.0,
            max_frame_bytes=int(tcp["max_frame_bytes"]),
        )

    def _accepts(self, request: Mapping[str, Any]) -> bool:
        tcp = self.settings["tcp"]
        accepted_functions = {str(x) for x in tcp.get("accepted_functions", []) if str(x)}
        accepted_cameras = {str(x) for x in tcp.get("accepted_cameras", []) if str(x)}
        accepted_task_ids = {str(x) for x in tcp.get("accepted_task_ids", []) if str(x)}
        function = str(request.get("function") or "")
        camera = str(request.get("camera") or "")
        task_id = str(request.get("task_id") or "")
        if function == "state":
            return False
        if accepted_functions and function not in accepted_functions:
            return False
        if accepted_cameras and camera not in accepted_cameras:
            return False
        if accepted_task_ids and task_id not in accepted_task_ids:
            return False
        return True

    @staticmethod
    def _cache_key(request: Mapping[str, Any]) -> str:
        return "|".join(
            [
                str(request.get("function") or ""),
                str(request.get("camera") or ""),
                str(request.get("task_id") or ""),
                str(request.get("triggerindex") or ""),
                str(request.get("triggerpos") or ""),
            ]
        )

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self.cache_lock:
            value = self.cache.get(key)
            if value is not None:
                self.cache.move_to_end(key)
                self.state.counters["duplicate_responses"] += 1
                return dict(value)
            return None

    def _cache_put(self, key: str, response: Mapping[str, Any]) -> None:
        limit = int(self.settings["tcp"].get("response_cache_size", 32))
        with self.cache_lock:
            self.cache[key] = dict(response)
            self.cache.move_to_end(key)
            while len(self.cache) > limit:
                self.cache.popitem(last=False)

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

    def _base_response(self, request: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = request.get("timestamp")
        if not (isinstance(timestamp, list) and len(timestamp) >= 2):
            timestamp = _timestamp_pair()
        return {
            "schema_version": "1.0",
            "message_type": "vision_detection_result",
            "function": str(self.settings["tcp"].get("response_function", "tube_pick_result")),
            # Echo request correlation fields as required by the scheduler protocol.
            "timestamp": [_int_value(timestamp[0]), _int_value(timestamp[1])],
            "response_timestamp": _timestamp_pair(),
            "triggerpos": _int_value(request.get("triggerpos"), _int_value(timestamp[0])),
            "triggerindex": _int_value(request.get("triggerindex"), 0),
            "camera": str(request.get("camera") or ""),
            "task_id": str(request.get("task_id") or ""),
            "camera_id": _int_value(self.settings["tcp"].get("camera_id", 0), 0),
            "barcodes": "",
            "distance": 0.0,
            "height": 0.0,
            # Existing VisionInterfacer treats empty types as an acknowledged result with no robot pose.
            "types": [],
            "poses": [],
        }

    def _error_response(self, request: Mapping[str, Any], code: int, label: str, error: Exception) -> dict[str, Any]:
        response = self._base_response(request)
        response.update(
            {
                "result": int(code),
                "status": "error",
                "result_text": label,
                "error": {"code": type(error).__name__, "message": str(error)},
                "coordinate_frame": "image_depth_aligned",
                "coordinate_units": {"x": "pixel", "y": "pixel", "z": "mm"},
                "product_detected": False,
                "separator_detected": False,
                "product_count": 0,
                "separator_count": 0,
                "products": [],
                "separators": [],
            }
        )
        return response

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if not self._accepts(request):
            self.state.counters["ignored_messages"] += 1
            return None
        key = self._cache_key(request)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not self.execution_lock.acquire(blocking=False):
            error = RuntimeError("视觉任务正在处理上一条触发")
            response = self._error_response(request, 1004, "vision_busy", error)
            self.state.counters["busy_rejections"] += 1
            return response
        self.state.begin(request)
        started = time.monotonic()
        try:
            runtime_result = self.runtime.infer_once()
            self._validate_runtime(runtime_result)
            classified = self.algorithm.classify(runtime_result)
            depth_bytes = b""
            depth = None
            depth_meta: dict[str, Any] = {}
            if classified.products:
                depth_bytes = self.http.get_bytes(self.depth_url)
                depth = decode_depth_png(depth_bytes)
                try:
                    depth_meta = self.http.request("GET", self.depth_meta_url).json()
                except Exception:
                    depth_meta = {}
                last_depth_ms = _int_value(depth_meta.get("last_depth_ms"), 0)
                if last_depth_ms > 0:
                    depth_age_ms = max(0, _timestamp_ms() - last_depth_ms)
                    if self.algorithm.max_age_ms and depth_age_ms > self.algorithm.max_age_ms:
                        raise ValueError(
                            f"深度帧过旧: age={depth_age_ms}ms, max={self.algorithm.max_age_ms}ms"
                        )
            payload, debug = self.algorithm.build_detection_payload(classified, depth)
            if classified.products:
                payload["depth"]["last_depth_ms"] = _int_value(depth_meta.get("last_depth_ms"), 0)
                payload["depth"]["age_ms"] = (
                    max(0, _timestamp_ms() - _int_value(depth_meta.get("last_depth_ms"), 0))
                    if _int_value(depth_meta.get("last_depth_ms"), 0) > 0 else None
                )
                debug["depth_meta"] = depth_meta
            invalid = int(payload["invalid_depth_count"])
            fail_on_invalid = self.algorithm.fail_on_invalid_depth
            response = self._base_response(request)
            response.update(payload)
            if invalid and fail_on_invalid:
                response.update({"result": 2, "status": "partial", "result_text": "product_depth_invalid"})
            else:
                response.update({"result": 0, "status": "ok", "result_text": "success"})
            response["frame_id"] = runtime_result.get("frame_id")
            response["result_id"] = runtime_result.get("result_id")
            response["model"] = runtime_result.get("model")
            response["timing"] = {"processing_ms": round((time.monotonic() - started) * 1000.0, 3)}
            debug_document = {
                "request": request,
                "response": response,
                "analysis": debug,
                "runtime_result": runtime_result,
            }
            self.state.success(response, debug_document)
            self._cache_put(key, response)
            self._save_debug_async(debug_document, depth_bytes)
            return response
        except Exception as error:
            code = 1001 if isinstance(error, UpstreamError) else 1002 if "深度" in str(error) else 1003
            response = self._error_response(request, code, "vision_processing_error", error)
            self.state.failure(response, error)
            self._cache_put(key, response)
            self._save_debug_async({"request": request, "response": response, "error": str(error)}, b"")
            return response
        finally:
            self.execution_lock.release()

    def evaluate_once(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request = dict(request or {})
        self.manual_trigger_index += 1
        timestamp = _timestamp_pair()
        request.setdefault("function", "manual_test")
        request.setdefault("timestamp", timestamp)
        request.setdefault("triggerpos", timestamp[0])
        request.setdefault("triggerindex", self.manual_trigger_index)
        request.setdefault("camera", "manual")
        request.setdefault("task_id", "tube_pick_vision")
        response = self.handle_message(request)
        if response is None:
            raise ValueError("手动请求被 accepted_* 过滤规则忽略")
        return response

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
        if not self.debug_enabled:
            return
        with self.debug_lock:
            self.debug_root.mkdir(parents=True, exist_ok=True)
            (self.debug_root / "request_response.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if depth_bytes:
                (self.debug_root / "depth.png").write_bytes(depth_bytes)
            try:
                runtime_result = (
                    document.get("runtime_result")
                    if isinstance(document.get("runtime_result"), Mapping)
                    else {}
                )
                rgb = self.runtime.snapshot()
                if rgb:
                    (self.debug_root / "rgb.jpg").write_bytes(rgb)
                    self._draw_overlay(
                        rgb,
                        runtime_result,
                        document.get("analysis"),
                        self.debug_root / "overlay.jpg",
                    )
            except Exception:
                pass

    @staticmethod
    def _draw_overlay(
        rgb_bytes: bytes,
        runtime_result: Mapping[str, Any],
        analysis: object,
        output_path: Path,
    ) -> None:
        image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        products_debug = []
        if isinstance(analysis, Mapping) and isinstance(analysis.get("products"), list):
            products_debug = analysis["products"]
        product_by_id = {str(item.get("id")): item for item in products_debug if isinstance(item, Mapping)}
        detections = runtime_result.get("detections") if isinstance(runtime_result.get("detections"), list) else []
        for item in detections:
            if not isinstance(item, Mapping):
                continue
            class_id = item.get("class_id")
            score = float(item.get("score") or 0.0)
            bbox = item.get("bbox_xyxy")
            center = item.get("center_xy")
            if class_id == 0 and isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if isinstance(center, list) and len(center) >= 2:
                    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
                    debug = product_by_id.get(str(item.get("id") or ""), {})
                    z = debug.get("z_mm")
                    text = f"product {score:.2f} ({cx},{cy},{z if z is not None else 'NA'}mm)"
                    cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif class_id == 1:
                # Separator position is visualized locally only and is not sent over TCP.
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    cv2.putText(image, f"separator {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        cv2.imwrite(str(output_path), image)

    def run(self) -> None:
        self.client.run(self.stop_event)

    def stop(self) -> None:
        self.stop_event.set()
        self.client.close()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "VisionOpsTubePick/1.0"

    @property
    def service(self) -> TubePickVisionService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, code: int, document: Mapping[str, Any]) -> None:
        # HTTP 调试接口返回标准 JSON，因此不会包含 TCP 传输层的 * / #。
        body = _json_bytes(document)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_tcp_frame(self, code: int, document: Mapping[str, Any]) -> None:
        # 使用与真实 TCP Client 完全相同的编码器返回 *<JSON>#，仅供联调检查。
        body = StarHashJsonCodec.encode(dict(document))
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
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
            snapshot = self.service.state.snapshot()
            self._send(200, {"ok": True, "status": snapshot["health"], "component": "tube_pick_vision"})
        elif path in {"/api/app/status", "/api/gateway/status", "/api/tcp/status"}:
            self._send(200, self.service.state.snapshot())
        elif path in {"/api/app/registers", "/api/gateway/registers"}:
            self._send(
                200,
                {
                    "schema_version": "1.0",
                    "message_type": "register_snapshot",
                    "status": "not_applicable",
                    "protocol": "tcp_json",
                    "registers": [],
                },
            )
        elif path in {"/api/app/latest_decision", "/api/app/latest_gateway_message"}:
            response = self.service.state.snapshot().get("latest_response")
            self._send(200, response or {"status": "empty", "message_type": "vision_detection_result"})
        else:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        json_paths = {"/api/app/evaluate_once", "/api/task/evaluate_once"}
        frame_path = "/api/tcp/evaluate_once_frame"
        if path not in json_paths | {frame_path}:
            self._send(404, {"status": "error", "error": {"code": "NOT_FOUND", "message": path}})
            return
        try:
            response = self.service.evaluate_once(self._read_json())
            if path == frame_path:
                self._send_tcp_frame(200, response)
            else:
                self._send(200, response)
        except Exception as error:
            document = {"status": "error", "error": {"code": type(error).__name__, "message": str(error)}}
            if path == frame_path:
                self._send_tcp_frame(500, document)
            else:
                self._send(500, document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps tube-pick TCP client service")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="carton_line unified YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    service = TubePickVisionService(config)
    http_config = config["pick"]["tcp"]["http"]
    server = ReusableThreadingHTTPServer(
        (str(http_config["listen_host"]), int(http_config["listen_port"])), StatusHandler
    )
    server.service = service  # type: ignore[attr-defined]
    stop_once = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        if stop_once.is_set():
            return
        stop_once.set()
        service.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    http_thread = threading.Thread(target=server.serve_forever, name="tube-pick-http", daemon=True)
    http_thread.start()
    print(
        "Tube Pick Vision started: "
        f"scheduler={config['pick']['tcp']['server_host']}:{config['pick']['tcp']['server_port']} "
        f"http={http_config['listen_host']}:{http_config['listen_port']} "
        f"runtime={config['runtimes']['pick']['url']}"
    )
    try:
        service.run()
    finally:
        service.stop()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
