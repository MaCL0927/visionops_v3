"""Runtime mock/RKNN backend 选择的无 SDK 集成测试。"""

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


def _request_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture(scope="session")
def backend_runtime_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("runtime-backend-build")
    subprocess.run(
        ["cmake", "-S", str(PROJECT_ROOT), "-B", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j4", "--target", "visionops_runtime_mock"],
        check=True,
        capture_output=True,
        text=True,
    )
    binary = build_dir / "edge/runtime_cpp/visionops_runtime_mock"
    assert binary.is_file()
    return binary


@contextmanager
def _running_runtime(binary: Path, backend: str | None):
    port = _free_port()
    command = [
        str(binary),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device-id",
        "example-backend-test",
    ]
    if backend is not None:
        command.extend(["--backend", backend])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Runtime 提前退出\nstdout={stdout}\nstderr={stderr}")
        try:
            _request_json(f"{base_url}/health")
            break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=3)
        pytest.fail("Runtime 未在超时时间内启动")

    try:
        yield base_url
    finally:
        process.terminate()
        process.wait(timeout=3)
        assert process.returncode == 0


@pytest.mark.parametrize("backend", [None, "mock"])
def test_mock_backend_remains_default_and_healthy(
    backend_runtime_binary: Path, backend: str | None
) -> None:
    with _running_runtime(backend_runtime_binary, backend) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert status["health"] == "ok"
        assert model["backend"] == "mock"
        assert model["runner_loaded"] is True
        assert model["rknn_compiled"] is False
        assert model["runner_error"] is None

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["task_type"] == "detection"
        assert result["detections"]
        assert "debug" not in result


def test_rknn_backend_degrades_cleanly_when_not_compiled(
    backend_runtime_binary: Path,
) -> None:
    with _running_runtime(backend_runtime_binary, "rknn") as base_url:
        health = _request_json(f"{base_url}/health")
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert health["health"] == "degraded"
        assert status["health"] == "degraded"
        assert model["backend"] == "rknn"
        assert model["runner_loaded"] is False
        assert model["rknn_compiled"] is False
        assert "未启用 RKNN" in model["runner_error"]
        assert "未启用 RKNN" in model["model_load_error"]

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["message_type"] == "inference_result"
        assert result["status"] == "error"
        assert result["error"]["code"] == "RKNN_MODEL_NOT_LOADED"
        assert result["debug"]["rknn_runner_called"] is False
        assert result["debug"]["raw_outputs_count"] == 0
        assert "未启用 RKNN" in result["error"]["message"]
