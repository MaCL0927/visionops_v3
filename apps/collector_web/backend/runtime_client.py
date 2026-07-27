"""Collector Web 到 Runtime/下游服务的受控 HTTP 客户端。

Collector 对本机 C++ Runtime 的高频代理请求优先使用 TCP_NODELAY 的原始
HTTP/1.1 客户端，避免 RK3576 上 urllib 对小 POST 请求产生约 20~30 ms 的固定
延迟。Gateway 与 Business App 默认仍保持原 urllib 路径，避免扩大本次改动范围。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_HEADER_BYTES = 128 * 1024


class RuntimeUnavailable(ConnectionError):
    """表示 Runtime 无法建立连接、请求超时或返回非法 HTTP 数据。"""


class _RawConnectUnavailable(RuntimeUnavailable):
    """原始本地 HTTP 在请求发送前无法建立连接，可安全回退 urllib。"""


@dataclass(frozen=True)
class RuntimeResponse:
    status_code: int
    content_type: str
    body: bytes
    headers: Mapping[str, str]
    transport: str = "urllib"
    elapsed_ms: float = 0.0

    def json(self) -> dict:
        value = json.loads(self.body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Runtime JSON 顶层必须是对象")
        return value


class RuntimeClient:
    """保留上游 HTTP 状态码和响应内容的轻量代理客户端。

    ``raw_local_enabled`` 只应为本机高频 Runtime 客户端开启。原始路径保持现有
    HTTP API、状态码、响应头和响应体不变；非 localhost URL 自动使用 urllib。
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 2.0,
        *,
        raw_local_enabled: bool = False,
        raw_local_fallback_urllib: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.raw_local_enabled = bool(raw_local_enabled)
        self.raw_local_fallback_urllib = bool(raw_local_fallback_urllib)
        self._stats_lock = threading.Lock()
        self._raw_request_count = 0
        self._raw_connect_failure_count = 0
        self._urllib_request_count = 0
        self._last_transport = "none"
        self._last_raw_error = ""

    @staticmethod
    def _supports_raw_local(url: str) -> bool:
        parsed = urlsplit(url)
        return parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }

    def _record_transport(self, transport: str, error: str = "") -> None:
        with self._stats_lock:
            self._last_transport = str(transport)
            if transport == "raw_socket":
                self._raw_request_count += 1
                self._last_raw_error = ""
            elif transport == "raw_connect_error":
                self._raw_connect_failure_count += 1
                self._last_raw_error = str(error)
            elif transport == "urllib":
                self._urllib_request_count += 1
                if error:
                    self._last_raw_error = str(error)

    def transport_status(self) -> dict[str, object]:
        with self._stats_lock:
            return {
                "raw_local_enabled": self.raw_local_enabled,
                "raw_local_fallback_urllib": self.raw_local_fallback_urllib,
                "raw_request_count": self._raw_request_count,
                "raw_connect_failure_count": self._raw_connect_failure_count,
                "urllib_request_count": self._urllib_request_count,
                "last_transport": self._last_transport,
                "last_raw_error": self._last_raw_error,
                "base_url": self.base_url,
            }

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> RuntimeResponse:
        url = f"{self.base_url}{path}"
        if self.raw_local_enabled and self._supports_raw_local(url):
            try:
                response = self._request_raw_local(
                    method=method,
                    url=url,
                    body=body,
                    content_type=content_type,
                )
                self._record_transport("raw_socket")
                return response
            except _RawConnectUnavailable as error:
                self._record_transport("raw_connect_error", str(error))
                if not self.raw_local_fallback_urllib:
                    raise
                return self._request_urllib(
                    method=method,
                    url=url,
                    body=body,
                    content_type=content_type,
                    raw_error=str(error),
                )
        return self._request_urllib(
            method=method,
            url=url,
            body=body,
            content_type=content_type,
        )

    def _request_urllib(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None,
        content_type: str,
        raw_error: str = "",
    ) -> RuntimeResponse:
        headers = {
            "Accept": "application/json, image/jpeg",
            "User-Agent": "visionops-collector-web/0.1",
        }
        if body is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers=headers,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = self._read_urllib_response(
                    response.status,
                    response.headers,
                    response,
                    started=started,
                )
        except urllib.error.HTTPError as error:
            result = self._read_urllib_response(
                error.code,
                error.headers,
                error,
                started=started,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise RuntimeUnavailable(str(reason)) from error
        self._record_transport("urllib", raw_error)
        return result

    def _read_urllib_response(
        self,
        status_code,
        headers,
        stream,
        *,
        started: float,
    ) -> RuntimeResponse:
        declared_length = headers.get("Content-Length")
        if declared_length is not None:
            try:
                content_length = int(declared_length)
            except ValueError as error:
                raise RuntimeUnavailable("Runtime Content-Length 非法") from error
            if content_length < 0 or content_length > MAX_RESPONSE_BYTES:
                raise RuntimeUnavailable("Runtime 响应超过 Collector 限制")
        body = stream.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeUnavailable("Runtime 响应超过 Collector 限制")
        normalized_headers = {key: value for key, value in headers.items()}
        content_type = headers.get_content_type() if hasattr(headers, "get_content_type") else ""
        return RuntimeResponse(
            status_code=int(status_code),
            content_type=content_type or "application/octet-stream",
            body=body,
            headers=normalized_headers,
            transport="urllib",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _request_raw_local(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None,
        content_type: str,
    ) -> RuntimeResponse:
        parsed = urlsplit(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        payload = body if body is not None else b""
        host_header = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        header_lines = [
            f"{method} {target} HTTP/1.1",
            f"Host: {host_header}",
            "Accept: application/json, image/jpeg",
            "User-Agent: visionops-collector-web-raw/1.0",
            "Connection: close",
            f"Content-Length: {len(payload)}",
        ]
        if body is not None:
            header_lines.append(f"Content-Type: {content_type}")
        request_bytes = (
            "\r\n".join(header_lines) + "\r\n\r\n"
        ).encode("ascii") + payload

        started = time.perf_counter()
        sock: socket.socket | None = None
        try:
            try:
                sock = socket.create_connection((host, port), timeout=self.timeout_s)
            except (OSError, TimeoutError) as error:
                raise _RawConnectUnavailable(str(error)) from error
            sock.settimeout(self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.sendall(request_bytes)
                header_bytes, initial_body = self._read_raw_headers(sock)
                status_code, headers = self._parse_raw_headers(header_bytes)
                response_body = self._read_raw_body(sock, headers, initial_body)
            except (OSError, TimeoutError, ValueError) as error:
                # POST 可能已经执行，响应阶段失败时不能再走 urllib 重试，避免重复推理。
                raise RuntimeUnavailable(f"Runtime raw HTTP 响应失败: {error}") from error
        finally:
            if sock is not None:
                sock.close()

        raw_content_type = headers.get("content-type", "application/octet-stream")
        normalized_content_type = raw_content_type.split(";", 1)[0].strip().lower()
        return RuntimeResponse(
            status_code=status_code,
            content_type=normalized_content_type or "application/octet-stream",
            body=response_body,
            headers=headers,
            transport="raw_socket",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _read_raw_headers(sock: socket.socket) -> tuple[bytes, bytes]:
        received = bytearray()
        while True:
            header_end = received.find(b"\r\n\r\n")
            if header_end >= 0:
                return bytes(received[:header_end]), bytes(received[header_end + 4 :])
            chunk = sock.recv(8192)
            if not chunk:
                raise ValueError("Runtime 在响应头完成前关闭连接")
            received.extend(chunk)
            if len(received) > MAX_RESPONSE_HEADER_BYTES:
                raise ValueError("Runtime 响应头超过 Collector 限制")

    @staticmethod
    def _parse_raw_headers(header_bytes: bytes) -> tuple[int, dict[str, str]]:
        text = header_bytes.decode("iso-8859-1")
        lines = text.split("\r\n")
        status_parts = lines[0].split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError("Runtime HTTP 状态行非法")
        status_code = int(status_parts[1])
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        return status_code, headers

    def _read_raw_body(
        self,
        sock: socket.socket,
        headers: Mapping[str, str],
        initial_body: bytes,
    ) -> bytes:
        transfer_encoding = headers.get("transfer-encoding", "").lower()
        if "chunked" in transfer_encoding:
            return self._read_chunked_body(sock, initial_body)

        declared_length = headers.get("content-length")
        if declared_length is not None:
            try:
                content_length = int(declared_length)
            except ValueError as error:
                raise ValueError("Runtime Content-Length 非法") from error
            if content_length < 0 or content_length > MAX_RESPONSE_BYTES:
                raise ValueError("Runtime 响应超过 Collector 限制")
            body = bytearray(initial_body)
            while len(body) < content_length:
                chunk = sock.recv(min(65536, content_length - len(body)))
                if not chunk:
                    raise ValueError("Runtime 在响应体完整前关闭连接")
                body.extend(chunk)
            return bytes(body[:content_length])

        body = bytearray(initial_body)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Runtime 响应超过 Collector 限制")
        return bytes(body)

    def _read_chunked_body(self, sock: socket.socket, initial_body: bytes) -> bytes:
        buffer = bytearray(initial_body)
        output = bytearray()

        def ensure_line() -> bytes:
            while True:
                marker = buffer.find(b"\r\n")
                if marker >= 0:
                    line = bytes(buffer[:marker])
                    del buffer[: marker + 2]
                    return line
                chunk = sock.recv(8192)
                if not chunk:
                    raise ValueError("Runtime chunked 响应提前结束")
                buffer.extend(chunk)

        while True:
            line = ensure_line()
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError as error:
                raise ValueError("Runtime chunk 大小非法") from error
            if size == 0:
                # 读取并忽略 trailer，直到空行。
                while ensure_line():
                    pass
                break
            if len(output) + size > MAX_RESPONSE_BYTES:
                raise ValueError("Runtime 响应超过 Collector 限制")
            while len(buffer) < size + 2:
                chunk = sock.recv(min(65536, size + 2 - len(buffer)))
                if not chunk:
                    raise ValueError("Runtime chunk 数据不完整")
                buffer.extend(chunk)
            output.extend(buffer[:size])
            if buffer[size : size + 2] != b"\r\n":
                raise ValueError("Runtime chunk 结尾非法")
            del buffer[: size + 2]
        return bytes(output)
