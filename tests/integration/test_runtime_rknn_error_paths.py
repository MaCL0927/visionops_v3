"""无 RKNN SDK 环境下的 Runtime 错误路径测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(url, data=b"{}" if method == "POST" else None, method=method)
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture(scope="session")
def rknn_error_binary(shared_runtime_binary: Path) -> Path:
    return shared_runtime_binary

@contextmanager
def _runtime(binary: Path, arguments: list[str]):
    port = _free_port()
    process = subprocess.Popen(
        [str(binary), "--host", "127.0.0.1", "--port", str(port), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            _json(f"{base_url}/health")
            break
        except OSError:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Runtime 提前退出\n{stdout}\n{stderr}")
            time.sleep(0.05)
    try:
        yield base_url
    finally:
        process.terminate()
        process.wait(timeout=3)


def test_rknn_not_compiled_returns_stable_error(rknn_error_binary: Path) -> None:
    with _runtime(rknn_error_binary, ["--backend", "rknn"]) as base_url:
        status = _json(f"{base_url}/api/runtime/status")
        assert status["health"] == "degraded"
        assert status["loaded_model"]["rknn_compiled"] is False
        result = _json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "error"
        assert result["error"]["code"] == "RKNN_MODEL_NOT_LOADED"
        assert result["error"]["recoverable"] is True
        assert _json(f"{base_url}/api/runtime/latest_result") == result


def test_missing_rknn_path_is_reported(
    rknn_error_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "missing_model"
    package.mkdir()
    (package / "model.yaml").write_text(
        "model_id: missing-model\ntask: detection\ninput_size: [640, 640]\nclass_names: [object]\n",
        encoding="utf-8",
    )
    with _runtime(
        rknn_error_binary,
        ["--backend", "rknn", "--model-dir", str(package)],
    ) as base_url:
        status = _json(f"{base_url}/api/runtime/status")
        error = status["loaded_model"]["model_load_error"]
        assert "缺少 model.rknn" in error
        assert "RKNN 模型文件不存在" in error
        assert status["loaded_model"]["rknn_path"].endswith("model.rknn")
