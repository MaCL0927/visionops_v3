"""Collector Web HTTP 响应构造工具。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Mapping


def timestamp_ms() -> int:
    import time

    return time.time_ns() // 1_000_000


def error_document(
    *,
    device_id: str,
    component: str,
    code: str,
    message: str,
    recoverable: bool,
    detail: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "message_type": "collector_error",
        "device_id": device_id,
        "component": component,
        "timestamp_ms": timestamp_ms(),
        "trace_id": f"trace-collector-error-{timestamp_ms()}",
        "source": "collector:http_api",
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "detail": detail,
            "recoverable": recoverable,
        },
    }


def send_bytes(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    body: bytes,
    content_type: str,
    headers: Mapping[str, str] | None = None,
) -> None:
    handler.send_response(status_code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    if headers:
        for name, value in headers.items():
            handler.send_header(name, value)
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def send_json(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    document: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> None:
    body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    send_bytes(handler, status_code, body, "application/json; charset=utf-8", headers)
