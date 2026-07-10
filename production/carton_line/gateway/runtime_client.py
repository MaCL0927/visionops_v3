"""HTTP clients for active Runtime inference and camera bridge frames."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping


class UpstreamError(ConnectionError):
    """Runtime or camera bridge request failed."""


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    content_type: str
    body: bytes
    headers: Mapping[str, str]

    def json(self) -> dict:
        try:
            value = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UpstreamError("上游返回的内容不是有效 JSON") from error
        if not isinstance(value, dict):
            raise UpstreamError("上游 JSON 顶层必须是对象")
        return value


class HttpClient:
    def __init__(self, timeout_s: float = 5.0, max_response_bytes: int = 32 * 1024 * 1024) -> None:
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes

    def request(self, method: str, url: str, body: bytes | None = None) -> HttpResponse:
        headers = {
            "Accept": "application/json,image/jpeg,image/png,*/*",
            "User-Agent": "visionops-v3-robot-gateway/1.0",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return self._read(response.status, response.headers, response)
        except urllib.error.HTTPError as error:
            response = self._read(error.code, error.headers, error)
            detail = response.body.decode("utf-8", errors="replace")[:500]
            raise UpstreamError(f"{method} {url} HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise UpstreamError(f"{method} {url} 失败: {reason}") from error

    def _read(self, status: int, headers, stream) -> HttpResponse:
        body = stream.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise UpstreamError("上游响应超过大小限制")
        content_type = headers.get_content_type() if hasattr(headers, "get_content_type") else ""
        return HttpResponse(int(status), content_type or "application/octet-stream", body, dict(headers.items()))

    def get_bytes(self, url: str) -> bytes:
        return self.request("GET", url).body


class RuntimeClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = HttpClient(timeout_s=timeout_s)

    def infer_once(self) -> dict:
        response = self.http.request("POST", f"{self.base_url}/api/runtime/infer_once", b"{}")
        result = response.json()
        if result.get("message_type") != "inference_result":
            raise UpstreamError("Runtime infer_once 未返回 inference_result")
        if result.get("status") != "ok":
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            raise UpstreamError(f"Runtime 推理失败: {error.get('code') or result.get('status')}")
        return result

    def status(self) -> dict:
        return self.http.request("GET", f"{self.base_url}/api/runtime/status").json()

    def snapshot(self) -> bytes:
        return self.http.get_bytes(f"{self.base_url}/api/runtime/snapshot.jpg")
