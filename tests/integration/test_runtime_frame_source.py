"""Runtime M10 帧源接口的无真实相机集成测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _request_json(url: str, method: str = "GET") -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.fixture(scope="session")
def runtime_frame_source_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("runtime-frame-source-build")
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
def _running_runtime(binary: Path, extra_args: list[str] | None = None):
    port = _free_port()
    command = [
        str(binary),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device-id",
        "frame-source-test",
    ]
    if extra_args:
        command.extend(extra_args)
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
            status, _ = _request_json(f"{base_url}/health")
            if status == 200:
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
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        assert process.returncode == 0


def test_default_mock_frame_source_is_reported(runtime_frame_source_binary: Path) -> None:
    with _running_runtime(runtime_frame_source_binary) as base_url:
        status_code, status = _request_json(f"{base_url}/api/runtime/status")
        assert status_code == 200
        assert status["frame_source"]["type"] == "mock"
        assert status["frame_source"]["opened"] is True
        assert status["camera_connected"] is True

        status_code, result = _request_json(
            f"{base_url}/api/runtime/infer_once", method="POST"
        )
        assert status_code == 200
        assert result["status"] == "ok"
        assert result["image"]["width"] == 1920
        assert result["image"]["height"] == 1080


def test_missing_test_image_returns_stable_json_error(
    runtime_frame_source_binary: Path,
) -> None:
    with _running_runtime(
        runtime_frame_source_binary,
        ["--frame-source", "test_image", "--test-image", "/tmp/visionops-missing.ppm"],
    ) as base_url:
        status_code, result = _request_json(
            f"{base_url}/api/runtime/infer_once", method="POST"
        )
        assert status_code == 200
        assert result["status"] == "error"
        assert result["error"]["code"] == "TEST_IMAGE_LOAD_FAILED"


def test_snapshot_stays_available_without_real_camera(
    runtime_frame_source_binary: Path,
) -> None:
    with _running_runtime(runtime_frame_source_binary) as base_url:
        with urllib.request.urlopen(f"{base_url}/api/runtime/snapshot.jpg", timeout=3) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers.get_content_type() == "image/jpeg"
        assert body.startswith(b"\xff\xd8")


def test_snapshot_uses_latest_rgb_frame_after_infer_once(
    runtime_frame_source_binary: Path,
) -> None:
    with _running_runtime(runtime_frame_source_binary) as base_url:
        status_code, result = _request_json(
            f"{base_url}/api/runtime/infer_once", method="POST"
        )
        assert status_code == 200
        assert result["status"] == "ok"

        with urllib.request.urlopen(f"{base_url}/api/runtime/snapshot.jpg", timeout=3) as response:
            body = response.read()
            assert response.status == 200
            assert response.headers.get_content_type() == "image/jpeg"
            assert response.headers.get("X-Frame-Id") == "frame-camera-1"
        assert body.startswith(b"\xff\xd8")
        assert body.endswith(b"\xff\xd9")
        assert len(body) > 4096

        head_request = urllib.request.Request(
            f"{base_url}/api/runtime/snapshot.jpg", method="HEAD"
        )
        with urllib.request.urlopen(head_request, timeout=3) as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "image/jpeg"
            assert int(response.headers["Content-Length"]) == len(body)
            assert response.read() == b""

        status_code, status = _request_json(f"{base_url}/api/runtime/status")
        assert status_code == 200
        assert status["frame_source"]["snapshot_encoder"] == "rgb888_jpeg"
        assert status["frame_source"]["latest_frame_id"] == "frame-camera-1"
        assert status["frame_source"]["frames_captured"] >= 1
