#!/usr/bin/env python3
"""HTTP business app for carton-palletizing first-layer placement."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from production.carton_palletizing.config import DEFAULT_CONFIG_PATH, load_config
from production.carton_palletizing.tasks.first_layer_placement.algorithm import FirstLayerPlacementAlgorithm


MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def timestamp_ms() -> int:
    return int(time.time() * 1000)


class UpstreamError(ConnectionError):
    pass


class RuntimeClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request_json(self, method: str, path: str, body: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8") if method == "POST" else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise UpstreamError(f"Runtime HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise UpstreamError(f"无法连接 Runtime: {getattr(error, 'reason', error)}") from error
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


class AppState:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.started_at = time.monotonic()
        self.lock = threading.RLock()
        self.latest_decision = None  # type: Optional[Dict[str, Any]]
        self.last_error = None  # type: Optional[Dict[str, Any]]
        self.counters = {"evaluate_attempts": 0, "evaluate_success": 0, "evaluate_failure": 0, "resets": 0}

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
            self.last_error = {
                "code": type(error).__name__,
                "message": str(error),
                "timestamp_ms": timestamp_ms(),
            }
            self.counters["evaluate_failure"] += 1

    def record_reset(self) -> None:
        with self.lock:
            self.latest_decision = None
            self.last_error = None
            self.counters["resets"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "schema_version": "1.0",
                "message_type": "app_status",
                "status": "ok",
                "health": "degraded" if self.last_error else "ok",
                "app_id": "first_layer_placement",
                "app_instance_id": "carton_palletizing-first-layer",
                "component": self.config["component"],
                "device_id": self.config["device_id"],
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(time.monotonic() - self.started_at, 3),
                "latest_decision": deepcopy(self.latest_decision),
                "latest_gateway_message": None,
                "register_snapshot": [],
                "counters": dict(self.counters),
                "last_error": deepcopy(self.last_error),
            }


class FirstLayerPlacementService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.algorithm = FirstLayerPlacementAlgorithm(config["task"]["algorithm"])
        timeout_s = float(config["app"]["request_timeout_ms"]) / 1000.0
        self.runtime = RuntimeClient(str(config["runtime"]["url"]), timeout_s)
        self.state = AppState(config)
        self.evaluate_lock = threading.Lock()
        self.allow_injected = bool(config.get("debug", {}).get("allow_injected_runtime_result", False))
        self.accepted_task_types = {
            str(item).strip().lower()
            for item in config.get("runtime", {}).get("accepted_task_types", [])
            if str(item).strip()
        }

    def _validate_runtime_result(self, runtime_result: Mapping[str, Any]) -> None:
        task_type = str(runtime_result.get("task_type") or "").strip().lower()
        if self.accepted_task_types and task_type not in self.accepted_task_types:
            raise ValueError(
                "纸箱摆放 Runtime 必须加载 OBB 模型，当前 task_type={!r}，允许值={}".format(
                    task_type, sorted(self.accepted_task_types)
                )
            )

    def reset(self) -> Dict[str, Any]:
        with self.evaluate_lock:
            self.algorithm.reset()
            self.state.record_reset()
        return {
            "schema_version": "1.0",
            "message_type": "app_command_result",
            "status": "ok",
            "command": "reset_first_layer",
            "timestamp_ms": timestamp_ms(),
        }

    def evaluate(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        with self.evaluate_lock:
            self.state.record_attempt()
            try:
                if bool(request_body.get("reset")):
                    self.algorithm.reset()
                    self.state.record_reset()
                injected = request_body.get("runtime_result")
                if isinstance(injected, Mapping):
                    if not self.allow_injected:
                        raise ValueError("当前配置不允许注入 runtime_result")
                    runtime_result = deepcopy(dict(injected))
                else:
                    runtime_result = self.runtime.infer_once()
                self._validate_runtime_result(runtime_result)
                placement = self.algorithm.evaluate(runtime_result)
                visualization_result = deepcopy(runtime_result)
                visualization_result["placement"] = placement
                decision = {
                    "schema_version": "1.0",
                    "message_type": "app_decision",
                    "status": "ok",
                    "app_id": "first_layer_placement",
                    "task": "first_layer_placement",
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
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

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
            status = service.state.snapshot()
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
            self._json(200, service.state.snapshot())
        elif path == "/api/app/registers":
            self._json(200, {"schema_version": "1.0", "message_type": "app_register_snapshot", "status": "ok", "registers": []})
        elif path == "/api/app/latest_decision":
            latest = service.state.snapshot()["latest_decision"]
            if latest is None:
                self._error(404, "LATEST_DECISION_NOT_FOUND", "尚未生成第一层摆放决策")
            else:
                self._json(200, latest)
        elif path == "/api/app/latest_gateway_message":
            self._error(404, "GATEWAY_NOT_IMPLEMENTED", "第一阶段尚未接入机器人 Gateway")
        elif path == "/api/gateway/status":
            self._json(200, {
                "schema_version": "1.0",
                "message_type": "gateway_status",
                "status": "not_configured",
                "health": "ok",
                "phase": 1,
                "reason": "第一阶段只实现 RGB 第一层摆放逻辑，尚未接入机器人协议",
            })
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
            elif path == "/api/app/reset":
                self._json(200, service.reset())
            else:
                self._error(404, "ROUTE_NOT_FOUND", "接口不存在")
        except ValueError as error:
            self._error(400, "INVALID_REQUEST", str(error))
        except UpstreamError as error:
            self._error(502, "RUNTIME_UNAVAILABLE", "纸箱摆放应用无法取得 Runtime 推理结果", str(error))
        except Exception as error:  # noqa: BLE001
            self._error(500, "EVALUATION_FAILED", "第一层摆放计算失败", str(error))


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
    print(
        f"Carton palletizing first-layer app listening on "
        f"{config['app']['listen_host']}:{config['app']['listen_port']}, Runtime={config['runtime']['url']}"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="纸箱托盘第一层摆放业务应用")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
