"""业务 Mock 输入和网络上游的统一加载器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from edge.gateway_adapter.result_fetcher import ResultFetcher


@dataclass(frozen=True)
class LoadedResult:
    status_code: int
    document: dict | None


class MockResultLoader:
    def __init__(
        self,
        *,
        upstream_kind: str,
        upstream_url: str,
        mock_case: str,
        mock_factories: Mapping[str, Callable[[], dict]],
    ) -> None:
        if upstream_kind not in {"collector", "runtime", "file"}:
            raise ValueError("upstream-kind 必须为 collector、runtime 或 file")
        if mock_case not in mock_factories:
            raise ValueError(f"不支持的 mock-case: {mock_case}")
        self.upstream_kind = upstream_kind
        self.upstream_url = upstream_url.rstrip("/")
        self.mock_case = mock_case
        self.mock_factories = dict(mock_factories)
        self.fetcher = (
            ResultFetcher(self.upstream_url, upstream_kind)
            if upstream_kind in {"collector", "runtime"}
            else None
        )

    def load(self) -> LoadedResult:
        if self.upstream_kind == "file":
            return LoadedResult(200, self.mock_factories[self.mock_case]())
        fetched = self.fetcher.fetch_latest_result()  # type: ignore[union-attr]
        return LoadedResult(fetched.status_code, fetched.document)
