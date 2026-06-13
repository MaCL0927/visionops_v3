"""Runtime M10 V4L2 错误路径测试，不依赖真实摄像头。"""

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
def runtime_camera_error_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("runtime-camera-error-build")
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
def _running_v4l2_missing_device(binary: Path):
    port = _free_port()
    command = [
        str(binary),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device-id",
        "camera-error-test",
        "--frame-source",
        "v4l2",
        "--camera-device",
        "/dev/visionops-missing-camera",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-pixel-format",
        "YUYV",
    ]
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


def test_v4l2_missing_device_does_not_crash(runtime_camera_error_binary: Path) -> None:
    with _running_v4l2_missing_device(runtime_camera_error_binary) as base_url:
        preview = _request_json(f"{base_url}/api/runtime/start_preview", method="POST")
        assert preview["health"] == "degraded"
        assert preview["camera_connected"] is False
        assert preview["frame_source"]["type"] == "v4l2"
        assert preview["frame_source"]["last_error"]

        second_preview = _request_json(f"{base_url}/api/runtime/start_preview", method="POST")
        assert second_preview["frame_source"]["type"] == "v4l2"

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "error"
        assert result["error"]["code"] == "CAMERA_FRAME_UNAVAILABLE"
        assert result["debug"]["frame_source_error"] is True

        stopped = _request_json(f"{base_url}/api/runtime/stop_preview", method="POST")
        assert stopped["mode"] == "idle"
        stopped_again = _request_json(f"{base_url}/api/runtime/stop_preview", method="POST")
        assert stopped_again["mode"] == "idle"
