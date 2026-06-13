"""从 Collector 或 Runtime 获取标准 inference_result。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class UpstreamUnavailable(ConnectionError):
    """上游连接失败或超时。"""


@dataclass(frozen=True)
class FetchResult:
    status_code: int
    document: dict | None


class ResultFetcher:
    def __init__(self, upstream_url: str, upstream_kind: str, timeout_s: float = 2.0) -> None:
        if upstream_kind not in {"collector", "runtime"}:
            raise ValueError("upstream_kind 必须为 collector 或 runtime")
        self.upstream_url = upstream_url.rstrip("/")
        self.upstream_kind = upstream_kind
        self.timeout_s = timeout_s

    @property
    def latest_result_url(self) -> str:
        return f"{self.upstream_url}/api/runtime/latest_result"

    def fetch_latest_result(self) -> FetchResult:
        request = urllib.request.Request(
            self.latest_result_url,
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "visionops-gateway-mock/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return FetchResult(response.status, self._read_json(response))
        except urllib.error.HTTPError as error:
            return FetchResult(error.code, self._read_json(error))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise UpstreamUnavailable(str(reason)) from error

    @staticmethod
    def _read_json(stream) -> dict:
        content_type = stream.headers.get_content_type()
        if content_type != "application/json":
            raise UpstreamUnavailable(f"上游返回非 JSON 内容: {content_type}")
        document = json.loads(stream.read(4 * 1024 * 1024).decode("utf-8"))
        if not isinstance(document, dict):
            raise UpstreamUnavailable("上游 JSON 顶层必须是对象")
        return document
