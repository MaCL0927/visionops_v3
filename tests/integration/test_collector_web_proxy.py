"""Collector Web 到 C++ Runtime Mock 的 HTTP 代理集成测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
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


def _request(url: str, method: str = "GET") -> tuple[int, str, bytes, dict[str, str]]:
    data = b"{}" if method == "POST" else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return (
                response.status,
                response.headers.get_content_type(),
                response.read(),
                dict(response.headers.items()),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            error.headers.get_content_type(),
            error.read(),
            dict(error.headers.items()),
        )


def _request_json(url: str, method: str = "GET") -> tuple[int, dict]:
    status, content_type, body, _ = _request(url, method)
    assert content_type == "application/json"
    return status, json.loads(body.decode("utf-8"))


def _wait_for_health(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"服务提前退出\nstdout={stdout}\nstderr={stderr}")
        try:
            status, _ = _request_json(url)
            if status == 200:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            time.sleep(0.05)
    pytest.fail(f"服务未在超时时间内启动: {url}")


@contextmanager
def _managed_process(command: list[str]):
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        if process.returncode != 0:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"服务退出码异常: {process.returncode}\n"
                f"command={command}\nstdout={stdout}\nstderr={stderr}"
            )


@pytest.fixture(scope="session")
def runtime_mock_binary_for_collector(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("collector-runtime-build")
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
    return build_dir / "edge/runtime_cpp/visionops_runtime_mock"


def _collector_command(
    port: int,
    runtime_url: str,
    gateway_url: str | None = None,
    business_app_url: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "apps.collector_web.backend.main",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--runtime-url",
        runtime_url,
        "--gateway-url",
        gateway_url or f"http://127.0.0.1:{_free_port()}",
        "--business-app-url",
        business_app_url or f"http://127.0.0.1:{_free_port()}",
        "--device-id",
        "example-edge-collector-test",
        "--component",
        "collector_web",
    ]


def test_collector_status_survives_unreachable_runtime() -> None:
    collector_port = _free_port()
    unavailable_runtime_port = _free_port()
    collector_url = f"http://127.0.0.1:{collector_port}"
    with _managed_process(
        _collector_command(
            collector_port,
            f"http://127.0.0.1:{unavailable_runtime_port}",
        )
    ) as collector:
        _wait_for_health(collector, f"{collector_url}/health")

        status, health = _request_json(f"{collector_url}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["component"] == "collector_web"

        status, combined = _request_json(f"{collector_url}/api/collector/status")
        assert status == 200
        assert combined["collector"]["status"] == "ok"
        assert combined["runtime"]["health"] == "unreachable"
        assert combined["runtime"]["reachable"] is False
        assert combined["runtime"]["error"]["code"] == "RUNTIME_UNREACHABLE"

        status, content_type, html, _ = _request(f"{collector_url}/")
        assert status == 200 and content_type == "text/html"
        for label in ("校验", "采集上传", "模型验证", "设置", "切换生产模式"):
            assert label.encode("utf-8") in html

        for path, expected_type in (
            ("/static/css/main.css", "text/css"),
            ("/static/js/main.js", "text/javascript"),
            ("/static/js/pages/calibration.js", "text/javascript"),
            ("/static/js/pages/capture.js", "text/javascript"),
            ("/static/js/pages/settings.js", "text/javascript"),
            ("/static/js/pages/production.js", "text/javascript"),
            ("/static/js/render/overlay.js", "text/javascript"),
        ):
            status, content_type, body, _ = _request(f"{collector_url}{path}")
            assert status == 200 and content_type == expected_type and body
            if path.endswith("production.js"):
                assert b"unreachable" in body

        status, gateway = _request_json(f"{collector_url}/api/gateway/status")
        assert status == 200
        assert gateway["status"] == "unreachable"
        assert gateway["reachable"] is False

        status, app = _request_json(f"{collector_url}/api/app/status")
        assert status == 200
        assert app["status"] == "unreachable"
        assert app["reachable"] is False

        status, frontend_config = _request_json(f"{collector_url}/api/collector/config")
        assert status == 200
        assert frontend_config["device_id"] == "example-edge-collector-test"

        frontend_root = PROJECT_ROOT / "apps/collector_web/frontend/static/js"
        source = "\n".join(path.read_text(encoding="utf-8") for path in frontend_root.rglob("*.js"))
        for direct_port in ("18080", "19090", "19110"):
            assert direct_port not in source
        assert 'fetch("http' not in source and "fetch('http" not in source


def test_collector_proxies_runtime_mock(
    runtime_mock_binary_for_collector: Path,
) -> None:
    runtime_port = _free_port()
    collector_port = _free_port()
    runtime_url = f"http://127.0.0.1:{runtime_port}"
    collector_url = f"http://127.0.0.1:{collector_port}"
    runtime_command = [
        str(runtime_mock_binary_for_collector),
        "--host",
        "127.0.0.1",
        "--port",
        str(runtime_port),
        "--device-id",
        "example-edge-collector-test",
        "--component",
        "rknn_runtime",
        "--mock-task-type",
        "detection",
    ]

    with _managed_process(runtime_command) as runtime:
        _wait_for_health(runtime, f"{runtime_url}/health")
        with _managed_process(_collector_command(collector_port, runtime_url)) as collector:
            _wait_for_health(collector, f"{collector_url}/health")

            status, missing = _request_json(f"{collector_url}/api/runtime/latest_result")
            assert status == 404
            assert missing["error"]["code"] == "LATEST_RESULT_NOT_FOUND"

            status, runtime_status = _request_json(f"{collector_url}/api/runtime/status")
            assert status == 200
            assert runtime_status["message_type"] == "runtime_status"
            assert runtime_status["mode"] == "idle"

            status, preview = _request_json(
                f"{collector_url}/api/runtime/start_preview", method="POST"
            )
            assert status == 200
            assert preview["running"] is True
            assert preview["mode"] == "preview"

            status, result = _request_json(
                f"{collector_url}/api/runtime/infer_once", method="POST"
            )
            assert status == 200
            assert result["message_type"] == "inference_result"
            assert result["frame_id"] == "frame-mock-00000001"

            status, latest = _request_json(f"{collector_url}/api/runtime/latest_result")
            assert status == 200
            assert latest == result

            status, content_type, snapshot, headers = _request(
                f"{collector_url}/api/runtime/snapshot.jpg"
            )
            assert status == 200
            assert content_type == "image/jpeg"
            assert headers["Cache-Control"] == "no-store"
            assert snapshot.startswith(b"\xff\xd8")
            assert snapshot.endswith(b"\xff\xd9")

            status, combined = _request_json(f"{collector_url}/api/collector/status")
            assert status == 200
            assert combined["runtime"]["reachable"] is True
            assert combined["runtime"]["health"] == "ok"


def test_collector_proxies_gateway_registers() -> None:
    collector_port = _free_port()
    gateway_port = _free_port()
    gateway_modbus_port = _free_port()
    unavailable_runtime_port = _free_port()
    unavailable_app_port = _free_port()
    gateway_command = [
        sys.executable,
        "-m",
        "edge.gateway_adapter.gateway_mock_service",
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        "--upstream-url",
        f"http://127.0.0.1:{unavailable_runtime_port}",
        "--upstream-kind",
        "runtime",
        "--modbus-host",
        "127.0.0.1",
        "--modbus-port",
        str(gateway_modbus_port),
        "--poll-interval-ms",
        "5000",
    ]
    collector_url = f"http://127.0.0.1:{collector_port}"
    with _managed_process(gateway_command) as gateway:
        _wait_for_health(gateway, f"http://127.0.0.1:{gateway_port}/health")
        command = _collector_command(
            collector_port,
            f"http://127.0.0.1:{unavailable_runtime_port}",
            f"http://127.0.0.1:{gateway_port}",
            f"http://127.0.0.1:{unavailable_app_port}",
        )
        with _managed_process(command) as collector:
            _wait_for_health(collector, f"{collector_url}/health")
            status, gateway_status = _request_json(f"{collector_url}/api/gateway/status")
            assert status == 200
            assert gateway_status["message_type"] == "gateway_status"
            status, registers = _request_json(f"{collector_url}/api/gateway/registers")
            assert status == 200
            assert registers["message_type"] == "holding_register_snapshot"
            assert len(registers["registers"]) == 20
