"""Runtime HTTP API 文档的最小契约测试。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "interfaces/protocols/runtime_http_api.md"
REQUIRED_ENDPOINTS = {
    "GET /health",
    "GET /api/runtime/status",
    "POST /api/runtime/start_preview",
    "POST /api/runtime/stop_preview",
    "POST /api/runtime/infer_once",
    "POST /api/runtime/switch_model",
    "GET /api/runtime/latest_result",
    "GET /api/runtime/snapshot.jpg",
}


def test_runtime_http_contract_contains_all_required_endpoints() -> None:
    content = CONTRACT_PATH.read_text(encoding="utf-8")
    missing = sorted(endpoint for endpoint in REQUIRED_ENDPOINTS if endpoint not in content)
    assert missing == []


def test_runtime_http_contract_documents_required_sections() -> None:
    content = CONTRACT_PATH.read_text(encoding="utf-8")
    for heading in ("### 用途", "### 请求参数", "### 成功响应", "### 错误状态", "### 调用关系"):
        assert content.count(heading) >= len(REQUIRED_ENDPOINTS)
