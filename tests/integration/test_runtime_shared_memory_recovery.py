"""Runtime shared-memory/HTTP fallback recovery regression tests."""

from __future__ import annotations

import http.server
import json
import mmap
import os
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


MAGIC = 0x564F505352474231
VERSION = 1
HEADER_SIZE = 192
BUFFER_COUNT = 2
PIXEL_FORMAT_RGB888 = 1
STATE_RUNNING = 1

JPEG_A = b"\xff\xd8VISIONOPS-FALLBACK-A\xff\xd9"
JPEG_B = b"\xff\xd8VISIONOPS-LIVE-B\xff\xd9"


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


class _BridgeState:
    jpeg = JPEG_A


class _BridgeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b'{"ok":true,"camera_connected":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path.startswith("/stream/snapshot.jpg"):
            body = _BridgeState.jpeg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
        else:
            body = b'{"ok":false}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _fake_bridge():
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _BridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def _shared_rgb(name: str, width: int = 2, height: int = 2):
    path = Path("/dev/shm") / name.lstrip("/")
    frame_capacity = width * height * 3
    total_size = HEADER_SIZE + frame_capacity * BUFFER_COUNT
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o660)
    os.ftruncate(fd, total_size)
    mapping = mmap.mmap(fd, total_size, access=mmap.ACCESS_WRITE)
    try:
        header = bytearray(HEADER_SIZE)
        struct.pack_into("<QIIQQQ", header, 0, MAGIC, VERSION, HEADER_SIZE, total_size, frame_capacity, frame_capacity)
        struct.pack_into(
            "<IIIIII", header, 40,
            width, height, 3, width * 3, PIXEL_FORMAT_RGB888, BUFFER_COUNT,
        )
        struct.pack_into("<IIQQQQQ", header, 64, STATE_RUNNING, 0, 1, int(time.time() * 1000), os.getpid(), 1, 0)
        mapping[:HEADER_SIZE] = header
        mapping[HEADER_SIZE:HEADER_SIZE + frame_capacity] = bytes([17]) * frame_capacity
        mapping.flush()
        yield mapping
    finally:
        mapping.close()
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _running_runtime(binary: Path, bridge_url: str, shm_name: str):
    port = _free_port()
    process = subprocess.Popen(
        [
            str(binary),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--device-id", "shm-recovery-test",
            "--frame-source", "shared_memory",
            "--shared-memory-name", shm_name,
            "--shared-memory-fallback-http", "true",
            "--hp60c-url", bridge_url,
            "--hp60c-snapshot-path", "/stream/snapshot.jpg",
            "--hp60c-health-path", "/health",
            "--camera-read-timeout-ms", "100",
            "--camera-reconnect-failure-threshold", "1",
            "--camera-reconnect-initial-ms", "20",
            "--camera-reconnect-max-ms", "50",
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
            pytest.fail(f"Runtime exited early\nstdout={stdout}\nstderr={stderr}")
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=2)
        pytest.fail("Runtime startup timeout")
    try:
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _post(url: str) -> bytes:
    request = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read()


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.read()


def _get_json(url: str) -> dict:
    return json.loads(_get(url).decode("utf-8"))


def test_shared_memory_recovery_does_not_pin_fallback_jpeg(shared_runtime_binary: Path) -> None:
    """A fallback JPEG must not remain frozen after shared RGB recovers."""
    shm_name = f"/visionops_test_rgb_{os.getpid()}_{time.time_ns()}"
    _BridgeState.jpeg = JPEG_A
    with _fake_bridge() as bridge_url:
        with _running_runtime(shared_runtime_binary, bridge_url, shm_name) as base_url:
            # Start preview while shared memory is absent.  The fallback path reads
            # JPEG_A into Runtime's local JPEG cache before RGB decode fails in the
            # no-OpenCV integration build.
            _post(f"{base_url}/api/runtime/start_preview")

            # Wait until the HTTP fallback has actually cached JPEG_A.  The no-OpenCV
            # test build reports a decode error only after storing that JPEG.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                status = _get_json(f"{base_url}/api/runtime/status")
                if "未启用 OpenCV" in str(status.get("frame_source", {}).get("last_error", "")):
                    break
                time.sleep(0.05)
            else:
                pytest.fail(f"HTTP fallback was not exercised: {status}")

            # Shared memory appears and supplies a fresh RGB frame.  The Bridge JPEG
            # simultaneously advances to JPEG_B, exactly like an Orbbec reconnect.
            with _shared_rgb(shm_name):
                _BridgeState.jpeg = JPEG_B
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    status = _get_json(f"{base_url}/api/runtime/status")
                    frame_source = status.get("frame_source", {})
                    if int(frame_source.get("shared_memory_sequence", 0)) >= 1:
                        break
                    time.sleep(0.05)
                else:
                    pytest.fail(f"Runtime did not recover shared memory: {status}")

                body = _get(f"{base_url}/api/runtime/snapshot.jpg?t={time.time_ns()}")
                assert body == JPEG_B, f"Runtime snapshot remained stale: {body!r}"

                # Repeated requests must continue to proxy the live Bridge cache,
                # not return the one JPEG cached during the outage.
                assert _get(f"{base_url}/api/runtime/snapshot.jpg?t={time.time_ns()}") == JPEG_B
