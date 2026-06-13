"""Collector Web 到 C++ Runtime 的受控 HTTP 客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping


MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class RuntimeUnavailable(ConnectionError):
    """表示 Runtime 无法建立连接或请求超时。"""


@dataclass(frozen=True)
class RuntimeResponse:
    status_code: int
    content_type: str
    body: bytes
    headers: Mapping[str, str]

    def json(self) -> dict:
        value = json.loads(self.body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Runtime JSON 顶层必须是对象")
        return value


class RuntimeClient:
    """保留 Runtime HTTP 状态码和响应内容的轻量代理客户端。"""

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> RuntimeResponse:
        headers = {
            "Accept": "application/json, image/jpeg",
            "User-Agent": "visionops-collector-web/0.1",
        }
        if body is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return self._read_response(response.status, response.headers, response)
        except urllib.error.HTTPError as error:
            return self._read_response(error.code, error.headers, error)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise RuntimeUnavailable(str(reason)) from error

    def _read_response(self, status_code, headers, stream) -> RuntimeResponse:
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
        )
