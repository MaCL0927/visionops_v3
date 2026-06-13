"""M6 两个业务 App Mock HTTP 集成测试。"""

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


ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, method: str = "GET") -> tuple[int, dict]:
    request = urllib.request.Request(url, data=b"{}" if method == "POST" else None, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


@contextmanager
def service(module: str, port: int, case: str, upstream_kind: str = "file", upstream_url: str = "http://127.0.0.1:8090"):
    process = subprocess.Popen([sys.executable, "-m", module, "--host", "127.0.0.1", "--port", str(port), "--upstream-kind", upstream_kind, "--upstream-url", upstream_url, "--mock-case", case, "--poll-interval-ms", "5000"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"服务提前退出\n{stdout}\n{stderr}")
            try:
                if request_json(f"http://127.0.0.1:{port}/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("业务 App 服务未就绪")
        yield f"http://127.0.0.1:{port}"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)
        if process.returncode != 0:
            stdout, stderr = process.communicate()
            pytest.fail(f"服务退出码异常: {process.returncode}\n{stdout}\n{stderr}")


@pytest.mark.parametrize(("module", "case", "label", "base"), [
    ("edge.gateway_adapter.apps.carton_tube_check.service", "ok", "OK", 100),
    ("edge.gateway_adapter.apps.carton_partition_check.service", "defect", "STRUCTURE_ABNORMAL", 200),
])
def test_business_file_mock_closed_loop(module: str, case: str, label: str, base: int) -> None:
    with service(module, free_port(), case) as url:
        status, decision = request_json(f"{url}/api/app/evaluate_once", "POST")
        assert status == 200 and decision["final_label"] == label
        assert request_json(f"{url}/api/app/latest_decision")[1] == decision
        registers = request_json(f"{url}/api/app/registers")[1]["registers"]
        assert len(registers) == 20 and registers[0]["address"] == base
        register_map = request_json(f"{url}/api/app/register_map")[1]["registers"]
        assert register_map[-1]["address"] == base + 19


def test_business_app_survives_unreachable_upstream() -> None:
    unavailable = free_port()
    with service("edge.gateway_adapter.apps.carton_tube_check.service", free_port(), "ok", "collector", f"http://127.0.0.1:{unavailable}") as url:
        status, error = request_json(f"{url}/api/app/evaluate_once", "POST")
        assert status == 502
        assert error["error"]["code"] == "UPSTREAM_UNREACHABLE"
        status, snapshot = request_json(f"{url}/api/app/status")
        assert status == 200 and snapshot["upstream"]["health"] == "unreachable"
