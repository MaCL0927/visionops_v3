"""Runtime、Collector、Gateway 与 Modbus Mock 的最小闭环测试。"""

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

from edge.modbus_adapter.modbus_test_client import ModbusTestClient


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


def _wait_health(process: subprocess.Popen, url: str) -> None:
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
    pytest.fail(f"服务未启动: {url}")


@contextmanager
def _process(command: list[str]):
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
                f"服务退出码异常: {process.returncode}\ncommand={command}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )


@pytest.fixture(scope="session")
def runtime_binary_for_gateway(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("gateway-runtime-build")
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


def test_gateway_stays_alive_when_upstream_unreachable() -> None:
    gateway_port = _free_port()
    modbus_port = _free_port()
    unavailable_port = _free_port()
    command = [
        sys.executable,
        "-m",
        "edge.gateway_adapter.gateway_mock_service",
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        "--upstream-url",
        f"http://127.0.0.1:{unavailable_port}",
        "--upstream-kind",
        "collector",
        "--modbus-host",
        "127.0.0.1",
        "--modbus-port",
        str(modbus_port),
        "--poll-interval-ms",
        "50",
    ]
    with _process(command) as gateway:
        base_url = f"http://127.0.0.1:{gateway_port}"
        _wait_health(gateway, f"{base_url}/health")
        time.sleep(0.15)
        status, document = _request_json(f"{base_url}/api/gateway/status")
        assert status == 200
        assert document["gateway"]["status"] == "ok"
        assert document["upstream"]["health"] == "unreachable"
        assert document["counters"]["upstream_errors"] >= 1


def test_runtime_collector_gateway_modbus_closed_loop(runtime_binary_for_gateway: Path) -> None:
    runtime_port = _free_port()
    collector_port = _free_port()
    gateway_port = _free_port()
    modbus_port = _free_port()
    runtime_url = f"http://127.0.0.1:{runtime_port}"
    collector_url = f"http://127.0.0.1:{collector_port}"
    gateway_url = f"http://127.0.0.1:{gateway_port}"

    runtime_command = [
        str(runtime_binary_for_gateway),
        "--host",
        "127.0.0.1",
        "--port",
        str(runtime_port),
        "--device-id",
        "example-edge-gateway-test",
        "--mock-task-type",
        "detection",
    ]
    collector_command = [
        sys.executable,
        "-m",
        "apps.collector_web.backend.main",
        "--host",
        "127.0.0.1",
        "--port",
        str(collector_port),
        "--runtime-url",
        runtime_url,
        "--device-id",
        "example-edge-gateway-test",
    ]
    gateway_command = [
        sys.executable,
        "-m",
        "edge.gateway_adapter.gateway_mock_service",
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        "--upstream-url",
        collector_url,
        "--upstream-kind",
        "collector",
        "--modbus-host",
        "127.0.0.1",
        "--modbus-port",
        str(modbus_port),
        "--poll-interval-ms",
        "5000",
        "--device-id",
        "example-edge-gateway-test",
        "--app-id",
        "generic_mock",
    ]

    with _process(runtime_command) as runtime:
        _wait_health(runtime, f"{runtime_url}/health")
        with _process(collector_command) as collector:
            _wait_health(collector, f"{collector_url}/health")
            with _process(gateway_command) as gateway:
                _wait_health(gateway, f"{gateway_url}/health")

                status, result = _request_json(
                    f"{runtime_url}/api/runtime/infer_once", method="POST"
                )
                assert status == 200

                status, message = _request_json(
                    f"{gateway_url}/api/gateway/poll_once", method="POST"
                )
                assert status == 200
                assert message["message_type"] == "gateway_message"
                assert message["result_id"] == result["result_id"]
                assert message["protocol"] == "modbus_tcp"

                status, latest = _request_json(f"{gateway_url}/api/gateway/latest_message")
                assert status == 200
                assert latest == message

                status, registers = _request_json(f"{gateway_url}/api/gateway/registers")
                assert status == 200
                assert len(registers["registers"]) == 20
                assert registers["registers"][7]["value"] == 1

                client = ModbusTestClient("127.0.0.1", modbus_port)
                values = client.read_holding_registers(0, 20)
                assert len(values) == 20
                assert values[0] == 1
                assert values[7] == 1
                assert values[8] == 940

                client.write_single_register(19, 1234)
                assert client.read_holding_registers(19, 1) == [1234]
                client.write_multiple_registers(18, [4321, 5678])
                assert client.read_holding_registers(18, 2) == [4321, 5678]
