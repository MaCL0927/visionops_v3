"""Low-overhead Runtime HTTP IPC with localhost raw-socket acceleration.

The C++ Runtime currently exposes an HTTP API.  Python's urllib path can add a
large fixed delay for tiny POST requests on some embedded Linux TCP stacks.
This module keeps the protocol unchanged while using one TCP_NODELAY send for
localhost requests.  It also returns the raw response bytes and detailed timing
so JSON decoding can be deferred to a separate CPU postprocess thread.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

from production.carton_line.gateway.runtime_client import UpstreamError

DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TimedHttpResponse:
    body: bytes
    status_code: int
    headers: Mapping[str, str]
    connect_ms: float
    send_ms: float
    headers_wait_ms: float
    body_read_ms: float
    total_ms: float
    transport: str

    def header_float(self, name: str) -> float:
        raw = self.headers.get(str(name).lower())
        try:
            return float(raw) if raw is not None else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0




class _RawLocalHttpClient:
    """One-shot localhost HTTP/1.1 client using TCP_NODELAY.

    Header and body are sent with one ``sendall`` call to avoid the fixed
    small-packet delay observed with urllib on the RK3576 deployment.
    """

    def __init__(self, timeout_s: float, max_response_bytes: int) -> None:
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)

    @staticmethod
    def supports(url: str) -> bool:
        parsed = urlsplit(url)
        return parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }

    def request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
    ) -> TimedHttpResponse:
        parsed = urlsplit(url)
        if not self.supports(url):
            raise ValueError("raw local HTTP 仅支持 localhost 的 http URL")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        payload = body or b""
        lines = [
            f"{method} {target} HTTP/1.1",
            f"Host: {host}:{port}",
            "Accept: application/json,image/jpeg,image/png,*/*",
            "User-Agent: visionops-runtime-ipc-raw/1.0",
            "Connection: close",
            f"Content-Length: {len(payload)}",
        ]
        if body is not None:
            lines.append("Content-Type: application/json")
        request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + payload

        started = time.perf_counter()
        sock: Optional[socket.socket] = None
        try:
            connect_started = time.perf_counter()
            sock = socket.create_connection((host, port), timeout=self.timeout_s)
            sock.settimeout(self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connected = time.perf_counter()
            sock.sendall(request)
            sent = time.perf_counter()

            received = bytearray()
            header_end = -1
            while header_end < 0:
                chunk = sock.recv(8192)
                if not chunk:
                    raise ConnectionError("Runtime 在响应头之前关闭连接")
                received.extend(chunk)
                if len(received) > 128 * 1024:
                    raise ConnectionError("Runtime 响应头过大")
                header_end = received.find(b"\r\n\r\n")
            headers_received = time.perf_counter()
            header_text = bytes(received[:header_end]).decode("iso-8859-1")
            body_buffer = bytearray(received[header_end + 4 :])
            header_lines = header_text.split("\r\n")
            status_parts = header_lines[0].split(" ", 2)
            if len(status_parts) < 2:
                raise ConnectionError("Runtime 返回了无效 HTTP 状态行")
            status_code = int(status_parts[1])
            headers: Dict[str, str] = {}
            for line in header_lines[1:]:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
            transfer_encoding = headers.get("transfer-encoding", "identity").lower()
            if transfer_encoding != "identity":
                raise ConnectionError("raw Runtime IPC 不支持 chunked 响应")
            content_length = int(headers.get("content-length", len(body_buffer)))
            if content_length < 0 or content_length > self.max_response_bytes:
                raise ConnectionError("Runtime 响应超过大小限制")
            while len(body_buffer) < content_length:
                chunk = sock.recv(min(65536, content_length - len(body_buffer)))
                if not chunk:
                    raise ConnectionError("Runtime 在响应体完整前关闭连接")
                body_buffer.extend(chunk)
            finished = time.perf_counter()
            return TimedHttpResponse(
                body=bytes(body_buffer[:content_length]),
                status_code=status_code,
                headers=headers,
                connect_ms=(connected - connect_started) * 1000.0,
                send_ms=(sent - connected) * 1000.0,
                headers_wait_ms=(headers_received - sent) * 1000.0,
                body_read_ms=(finished - headers_received) * 1000.0,
                total_ms=(finished - started) * 1000.0,
                transport="raw_socket",
            )
        finally:
            if sock is not None:
                sock.close()


class RuntimeIpcClient:
    """Runtime client optimized for a Runtime on the same vision box.

    The raw path keeps the existing HTTP endpoint and response schema.  When it
    is unavailable, the client can fall back to urllib so deployment remains
    compatible with remote Runtime URLs and older environments.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        config = settings if isinstance(settings, Mapping) else {}
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = max(
            1024,
            int(config.get("max_response_bytes", DEFAULT_MAX_RESPONSE_BYTES)),
        )
        self.raw_http_enabled = bool(config.get("raw_http_enabled", True))
        self.raw_http_fallback_urllib = bool(
            config.get("raw_http_fallback_urllib", True)
        )
        self.raw_client = _RawLocalHttpClient(
            timeout_s=self.timeout_s,
            max_response_bytes=self.max_response_bytes,
        )
        self._stats_lock = threading.Lock()
        self._raw_request_count = 0
        self._raw_failure_count = 0
        self._urllib_request_count = 0
        self._last_transport = "none"
        self._last_raw_error = ""

    def _record(self, transport: str, raw_error: str = "") -> None:
        with self._stats_lock:
            self._last_transport = str(transport)
            if transport == "raw_socket":
                self._raw_request_count += 1
                self._last_raw_error = ""
            elif transport == "raw_error":
                self._raw_failure_count += 1
                self._last_raw_error = str(raw_error)
            elif transport == "urllib":
                self._urllib_request_count += 1
                if raw_error:
                    self._last_raw_error = str(raw_error)

    def transport_status(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "raw_http_enabled": self.raw_http_enabled,
                "raw_http_fallback_urllib": self.raw_http_fallback_urllib,
                "max_response_bytes": self.max_response_bytes,
                "raw_request_count": self._raw_request_count,
                "raw_failure_count": self._raw_failure_count,
                "urllib_request_count": self._urllib_request_count,
                "last_transport": self._last_transport,
                "last_raw_error": self._last_raw_error,
                "base_url": self.base_url,
            }

    def _urllib_request(
        self,
        method: str,
        url: str,
        body: Optional[bytes],
        raw_error: str = "",
    ) -> TimedHttpResponse:
        headers = {
            "Accept": "application/json,image/jpeg,image/png,*/*",
            "User-Agent": "visionops-detergent-runtime-ipc/1.0",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=body,
            method=str(method),
            headers=headers,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                headers_received = time.perf_counter()
                payload = response.read(self.max_response_bytes + 1)
                finished = time.perf_counter()
                status_code = int(getattr(response, "status", 200))
                normalized_headers = {
                    str(key).lower(): str(value)
                    for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise UpstreamError(
                f"{method} {url} HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise UpstreamError(f"{method} {url} 失败: {reason}") from error
        if len(payload) > self.max_response_bytes:
            raise UpstreamError("Runtime 响应超过大小限制")
        self._record("urllib", raw_error)
        return TimedHttpResponse(
            body=payload,
            status_code=status_code,
            headers=normalized_headers,
            connect_ms=0.0,
            send_ms=0.0,
            headers_wait_ms=(headers_received - started) * 1000.0,
            body_read_ms=(finished - headers_received) * 1000.0,
            total_ms=(finished - started) * 1000.0,
            transport="urllib",
        )

    def request_raw(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
    ) -> TimedHttpResponse:
        raw_error = ""
        if self.raw_http_enabled and self.raw_client.supports(url):
            try:
                response = self.raw_client.request(str(method), url, body)
                if response.status_code >= 400:
                    detail = response.body[:1000].decode(
                        "utf-8", errors="replace"
                    )
                    raise UpstreamError(
                        f"{method} {url} HTTP {response.status_code}: {detail}"
                    )
                self._record("raw_socket")
                return TimedHttpResponse(
                    body=response.body,
                    status_code=response.status_code,
                    headers=response.headers,
                    connect_ms=response.connect_ms,
                    send_ms=response.send_ms,
                    headers_wait_ms=response.headers_wait_ms,
                    body_read_ms=response.body_read_ms,
                    total_ms=response.total_ms,
                    transport=response.transport,
                )
            except UpstreamError:
                raise
            except (OSError, ValueError, ConnectionError, TimeoutError) as error:
                raw_error = str(error)
                self._record("raw_error", raw_error)
                if not self.raw_http_fallback_urllib:
                    raise UpstreamError(
                        f"{method} {url} raw HTTP 失败: {error}"
                    ) from error
        return self._urllib_request(method, url, body, raw_error=raw_error)

    @staticmethod
    def decode_json(raw: bytes) -> Dict[str, Any]:
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("Runtime 返回的内容不是有效 JSON") from error
        if not isinstance(document, dict):
            raise UpstreamError("Runtime JSON 顶层必须是对象")
        return document

    @classmethod
    def decode_inference(cls, raw: bytes) -> Dict[str, Any]:
        result = cls.decode_json(raw)
        if result.get("message_type") != "inference_result":
            raise UpstreamError("Runtime infer_once 未返回 inference_result")
        if result.get("status") != "ok":
            error = result.get("error") if isinstance(result.get("error"), Mapping) else {}
            raise UpstreamError(
                f"Runtime 推理失败: {error.get('code') or result.get('status')}"
            )
        return result

    def infer_once_raw(self) -> TimedHttpResponse:
        return self.request_raw(
            "POST",
            self.base_url + "/api/runtime/infer_once",
            b"{}",
        )

    def infer_once(self) -> Dict[str, Any]:
        return self.decode_inference(self.infer_once_raw().body)

    def status(self) -> Dict[str, Any]:
        response = self.request_raw(
            "GET",
            self.base_url + "/api/runtime/status",
        )
        return self.decode_json(response.body)

    def snapshot(self) -> bytes:
        return self.request_raw(
            "GET",
            self.base_url + "/api/runtime/snapshot.jpg",
        ).body
