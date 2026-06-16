#!/usr/bin/env python3
"""VisionOps v3 Collector Web 最小后端与 Runtime HTTP 代理。"""

from __future__ import annotations

import json
import mimetypes
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config_loader import CollectorConfig, load_config
from .response_utils import error_document, send_bytes, send_json, timestamp_ms
from .runtime_client import RuntimeClient, RuntimeResponse, RuntimeUnavailable


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
MAX_REQUEST_BODY_BYTES = 1024 * 1024
PROXY_PATHS = {
    "/api/runtime/status": "GET",
    "/api/runtime/start_preview": "POST",
    "/api/runtime/stop_preview": "POST",
    "/api/runtime/infer_once": "POST",
    "/api/runtime/latest_result": "GET",
    "/api/runtime/snapshot.jpg": "GET",
}
DOWNSTREAM_PATHS = {
    "/api/gateway/status": ("gateway", "/api/gateway/status", True),
    "/api/gateway/registers": ("gateway", "/api/gateway/registers", False),
    "/api/app/status": ("business_app", "/api/app/status", True),
    "/api/app/registers": ("business_app", "/api/app/registers", False),
    "/api/app/latest_decision": ("business_app", "/api/app/latest_decision", False),
    "/api/app/latest_gateway_message": ("business_app", "/api/app/latest_gateway_message", False),
}


class CollectorServer(ThreadingHTTPServer):
    """保存 Collector 运行上下文的线程化 HTTP 服务。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__((config.host, config.port), CollectorRequestHandler)
        self.config = config
        self.started_at = time.monotonic()
        self.runtime_client = RuntimeClient(config.runtime_url)
        self.gateway_client = RuntimeClient(config.gateway_url)
        self.business_app_client = RuntimeClient(config.business_app_url)

    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at


class CollectorRequestHandler(BaseHTTPRequestHandler):
    """只提供静态页面、Collector 状态和 Runtime HTTP 代理。"""

    server: CollectorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self._serve_static(path)
            return
        if path == "/health":
            self._send_health()
            return
        if path == "/api/collector/status":
            self._send_collector_status()
            return
        if path == "/api/collector/config":
            self._send_frontend_config()
            return
        if path in DOWNSTREAM_PATHS:
            name, target, status_endpoint = DOWNSTREAM_PATHS[path]
            self._proxy_downstream(name, target, status_endpoint)
            return
        if path in PROXY_PATHS:
            self._proxy_runtime(path, expected_method=PROXY_PATHS[path])
            return
        self._send_collector_error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/app/evaluate_once":
            body = self._read_request_body()
            if body is None:
                return
            self._proxy_downstream_post("business_app", "/api/app/evaluate_once", body)
            return
        if path in PROXY_PATHS:
            self._proxy_runtime(path, expected_method=PROXY_PATHS[path])
            return
        self._send_collector_error(404, "ROUTE_NOT_FOUND", "接口不存在", True)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_collector_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在", False)
            return
        send_bytes(self, 200, body, content_type, {"Cache-Control": "no-cache"})

    def _serve_static(self, request_path: str) -> None:
        relative = request_path[len("/static/"):] if request_path.startswith("/static/") else request_path.lstrip("/")
        if not relative or ".." in Path(relative).parts:
            self._send_collector_error(404, "STATIC_FILE_NOT_FOUND", "静态资源不存在", False)
            return
        target = FRONTEND_DIR / "static" / relative
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type in {"text/javascript", "text/css"}:
            content_type += "; charset=utf-8"
        self._serve_file(target, content_type)

    def _send_health(self) -> None:
        config = self.server.config
        send_json(
            self,
            200,
            {
                "schema_version": "1.0",
                "message_type": "collector_health",
                "status": "ok",
                "component": config.component,
                "device_id": config.device_id,
                "timestamp_ms": timestamp_ms(),
                "uptime_s": round(self.server.uptime_s(), 3),
                "runtime_url": config.runtime_url,
                "gateway_url": config.gateway_url,
                "business_app_url": config.business_app_url,
            },
        )

    def _send_frontend_config(self) -> None:
        config = self.server.config
        send_json(self, 200, {
            "schema_version": "1.0",
            "message_type": "collector_frontend_config",
            "runtime_url": config.runtime_url,
            "gateway_url": config.gateway_url,
            "business_app_url": config.business_app_url,
            "device_id": config.device_id,
            "snapshot_refresh_interval_ms": config.snapshot_refresh_interval_ms,
            "status_refresh_interval_ms": config.status_refresh_interval_ms,
        })

    def _collector_snapshot(self) -> dict[str, Any]:
        config = self.server.config
        return {
            "status": "ok",
            "component": config.component,
            "device_id": config.device_id,
            "uptime_s": round(self.server.uptime_s(), 3),
        }

    def _send_collector_status(self) -> None:
        runtime: dict[str, Any]
        try:
            health_response = self.server.runtime_client.request("GET", "/health")
            status_response = self.server.runtime_client.request("GET", "/api/runtime/status")
            runtime = {
                "health": "ok" if health_response.status_code == 200 else "error",
                "reachable": True,
                "health_status_code": health_response.status_code,
                "status_status_code": status_response.status_code,
                "health_response": self._decode_runtime_json(health_response),
                "status_response": self._decode_runtime_json(status_response),
            }
        except (RuntimeUnavailable, ValueError, json.JSONDecodeError) as error:
            runtime = {
                "health": "unreachable",
                "reachable": False,
                "error": {
                    "code": "RUNTIME_UNREACHABLE",
                    "message": "Collector 无法连接 Runtime",
                    "detail": str(error),
                    "recoverable": True,
                },
            }

        send_json(
            self,
            200,
            {
                "schema_version": "1.0",
                "message_type": "collector_status",
                "timestamp_ms": timestamp_ms(),
                "collector": self._collector_snapshot(),
                "runtime": runtime,
                "proxy": {
                    "runtime_url": self.server.config.runtime_url,
                    "gateway_url": self.server.config.gateway_url,
                    "business_app_url": self.server.config.business_app_url,
                    "timeout_s": self.server.runtime_client.timeout_s,
                    "mode": "http",
                },
            },
        )

    def _proxy_downstream(self, name: str, target: str, status_endpoint: bool) -> None:
        clients = {
            "gateway": (self.server.gateway_client, self.server.config.gateway_url),
            "business_app": (self.server.business_app_client, self.server.config.business_app_url),
        }
        client, service_url = clients[name]
        try:
            response = client.request("GET", target)
        except RuntimeUnavailable as error:
            if status_endpoint:
                send_json(self, 200, {
                    "schema_version": "1.0",
                    "message_type": f"{name}_proxy_status",
                    "status": "unreachable",
                    "health": "unreachable",
                    "reachable": False,
                    "service": name,
                    "error": {
                        "code": f"{name.upper()}_UNREACHABLE",
                        "message": f"Collector 无法连接 {name}",
                        "detail": str(error),
                        "recoverable": True,
                    },
                })
            else:
                self._send_collector_error(
                    502, f"{name.upper()}_UNREACHABLE", f"Collector 无法连接 {name}",
                    True, detail=str(error),
                )
            return
        if response.content_type != "application/json":
            self._send_collector_error(502, "INVALID_DOWNSTREAM_RESPONSE", "下游返回非 JSON 内容", True, detail={"service": name, "content_type": response.content_type})
            return
        send_bytes(self, response.status_code, response.body, "application/json; charset=utf-8", {
            "X-VisionOps-Proxied-By": self.server.config.component,
            "X-VisionOps-Downstream-Url": service_url,
        })


    def _proxy_downstream_post(self, name: str, target: str, body: bytes) -> None:
        clients = {
            "gateway": (self.server.gateway_client, self.server.config.gateway_url),
            "business_app": (self.server.business_app_client, self.server.config.business_app_url),
        }
        client, service_url = clients[name]
        try:
            response = client.request("POST", target, body=body)
        except RuntimeUnavailable as error:
            self._send_collector_error(
                502, f"{name.upper()}_UNREACHABLE", f"Collector 无法连接 {name}",
                True, detail=str(error),
            )
            return
        if response.content_type != "application/json":
            self._send_collector_error(
                502, "INVALID_DOWNSTREAM_RESPONSE", "下游返回非 JSON 内容",
                True, detail={"service": name, "content_type": response.content_type},
            )
            return
        send_bytes(self, response.status_code, response.body, "application/json; charset=utf-8", {
            "X-VisionOps-Proxied-By": self.server.config.component,
            "X-VisionOps-Downstream-Url": service_url,
        })

    def _decode_runtime_json(self, response: RuntimeResponse) -> dict[str, Any]:
        if response.content_type != "application/json":
            raise ValueError(f"Runtime 返回非 JSON 内容: {response.content_type}")
        return response.json()

    def _read_request_body(self) -> bytes | None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_collector_error(400, "INVALID_CONTENT_LENGTH", "Content-Length 非法", True)
            return None
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            self._send_collector_error(413, "REQUEST_BODY_TOO_LARGE", "请求体超过限制", True)
            return None
        return self.rfile.read(length) if length else b"{}"

    def _proxy_runtime(self, path: str, expected_method: str) -> None:
        if self.command != expected_method:
            self._send_collector_error(
                405,
                "METHOD_NOT_ALLOWED",
                f"请求方法不支持，期望 {expected_method}",
                True,
                headers={"Allow": expected_method},
            )
            return

        body = self._read_request_body() if self.command == "POST" else None
        if self.command == "POST" and body is None:
            return
        target = path
        query = urlsplit(self.path).query
        if query:
            target = f"{target}?{query}"
        try:
            response = self.server.runtime_client.request(self.command, target, body=body)
        except RuntimeUnavailable as error:
            self._send_collector_error(
                502,
                "RUNTIME_UNREACHABLE",
                "Collector 无法连接 Runtime",
                True,
                detail=str(error),
            )
            return

        if path == "/api/runtime/snapshot.jpg" and response.content_type == "image/jpeg":
            forwarded_headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower() in {"cache-control", "x-frame-id", "x-trace-id", "x-timestamp-ms"}
            }
            send_bytes(self, response.status_code, response.body, "image/jpeg", forwarded_headers)
            return

        if response.content_type != "application/json":
            self._send_collector_error(
                502,
                "INVALID_RUNTIME_RESPONSE",
                "Runtime 返回了非预期内容类型",
                True,
                detail={"content_type": response.content_type},
            )
            return
        send_bytes(
            self,
            response.status_code,
            response.body,
            "application/json; charset=utf-8",
            {
                "X-VisionOps-Proxied-By": self.server.config.component,
                "X-VisionOps-Runtime-Url": self.server.config.runtime_url,
                "X-VisionOps-Proxy-Timestamp-Ms": str(timestamp_ms()),
            },
        )

    def _send_collector_error(
        self,
        status_code: int,
        code: str,
        message: str,
        recoverable: bool,
        detail: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        send_json(
            self,
            status_code,
            error_document(
                device_id=self.server.config.device_id,
                component=self.server.config.component,
                code=code,
                message=message,
                recoverable=recoverable,
                detail=detail,
            ),
            headers,
        )


def run(config: CollectorConfig) -> int:
    server = CollectorServer(config)
    stop_requested = threading.Event()

    def request_shutdown(_signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            stop_requested.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        f"VisionOps Collector Web 正在监听 {config.host}:{config.port}，"
        f"Runtime={config.runtime_url}，Gateway={config.gateway_url}，App={config.business_app_url}"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    print("VisionOps Collector Web 已停止")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(load_config(argv))


if __name__ == "__main__":
    raise SystemExit(main())
