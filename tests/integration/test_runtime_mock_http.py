"""C++ Runtime Mock HTTP 服务的无设备集成测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.interfaces.validate_interface_examples import validate_example


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _request_json(url: str, method: str = "GET") -> tuple[int, dict]:
    data = b"{}" if method == "POST" else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.fixture(scope="session")
def runtime_mock_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("runtime-mock-build")
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


@pytest.fixture
def runtime_server(runtime_mock_binary: Path):
    port = _free_port()
    process = subprocess.Popen(
        [
            str(runtime_mock_binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device-id",
            "example-edge-integration",
            "--component",
            "rknn_runtime",
            "--mock-task-type",
            "detection",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Runtime Mock 提前退出\nstdout={stdout}\nstderr={stderr}")
        try:
            status, _ = _request_json(f"{base_url}/health")
            if status == 200:
                break
        except (OSError, ValueError):
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=3)
        pytest.fail("Runtime Mock 未在超时时间内启动")

    yield base_url

    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    assert process.returncode == 0


def test_runtime_mock_state_and_inference_flow(runtime_server: str) -> None:
    status_code, health = _request_json(f"{runtime_server}/health")
    assert status_code == 200
    assert health["status"] == "ok"
    assert health["device_id"] == "example-edge-integration"

    status_code, initial = _request_json(f"{runtime_server}/api/runtime/status")
    assert status_code == 200
    assert initial["running"] is False
    assert initial["mode"] == "idle"
    assert initial["counters"]["frames_in"] == 0
    validate_example(initial, "runtime status response")

    status_code, missing = _request_json(f"{runtime_server}/api/runtime/latest_result")
    assert status_code == 404
    assert missing["status"] == "error"
    assert missing["error"]["code"] == "LATEST_RESULT_NOT_FOUND"
    assert missing["error"]["recoverable"] is True

    status_code, preview = _request_json(
        f"{runtime_server}/api/runtime/start_preview", method="POST"
    )
    assert status_code == 200
    assert preview["running"] is True
    assert preview["mode"] == "preview"

    status_code, first_result = _request_json(
        f"{runtime_server}/api/runtime/infer_once", method="POST"
    )
    assert status_code == 200
    assert first_result["message_type"] == "inference_result"
    assert first_result["task_type"] == "detection"
    assert first_result["frame_id"] == "frame-mock-00000001"
    assert first_result["result_id"] == "result-mock-00000001"
    validate_example(first_result, "first inference response")

    status_code, second_result = _request_json(
        f"{runtime_server}/api/runtime/infer_once", method="POST"
    )
    assert status_code == 200
    assert second_result["frame_id"] == "frame-mock-00000002"
    assert second_result["result_id"] == "result-mock-00000002"
    validate_example(second_result, "second inference response")

    status_code, latest = _request_json(f"{runtime_server}/api/runtime/latest_result")
    assert status_code == 200
    assert latest == second_result

    status_code, current = _request_json(f"{runtime_server}/api/runtime/status")
    assert status_code == 200
    assert current["running"] is True
    assert current["mode"] == "detect"
    assert current["counters"]["frames_in"] == 2
    assert current["counters"]["frames_inferred"] == 2
    assert current["last_frame_id"] == second_result["frame_id"]
    assert current["last_result_id"] == second_result["result_id"]

    status_code, stopped = _request_json(
        f"{runtime_server}/api/runtime/stop_preview", method="POST"
    )
    assert status_code == 200
    assert stopped["running"] is False
    assert stopped["mode"] == "idle"


def test_runtime_mock_snapshot_is_embedded_jpeg(runtime_server: str) -> None:
    with urllib.request.urlopen(f"{runtime_server}/api/runtime/snapshot.jpg", timeout=3) as response:
        body = response.read()
        assert response.status == 200
        assert response.headers.get_content_type() == "image/jpeg"
        assert response.headers["Cache-Control"] == "no-store"
    assert body.startswith(b"\xff\xd8")
    assert body.endswith(b"\xff\xd9")
