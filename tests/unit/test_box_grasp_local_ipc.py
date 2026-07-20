from __future__ import annotations

import json
import mmap
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from production.carton_palletizing.tasks.box_grasp_vision.local_ipc import (
    SHARED_DEPTH_HEADER,
    SHARED_DEPTH_HEADER_SIZE,
    SHARED_DEPTH_MAGIC,
    RawLocalHttpClient,
    SharedDepthReader,
)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.dumps({"ok": True}, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def test_raw_local_http_client_reads_content_length_response():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RawLocalHttpClient(timeout_s=2.0, max_response_bytes=4096)
        response = client.request("POST", f"http://127.0.0.1:{server.server_port}/infer", b"{}")
        assert response.status_code == 200
        assert json.loads(response.body) == {"ok": True}
        assert response.transport == "raw_socket"
        assert response.total_ms >= response.headers_wait_ms
        assert response.body_read_ms >= 0.0
    finally:
        server.shutdown()
        server.server_close()


def test_shared_depth_reader_samples_without_full_frame_copy():
    name = f"/visionops_test_depth_{os.getpid()}"
    path = "/dev/shm/" + name.lstrip("/")
    width, height = 8, 6
    capacity = width * height * 2
    total_size = SHARED_DEPTH_HEADER_SIZE + 2 * capacity
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    os.ftruncate(fd, total_size)
    mapping = mmap.mmap(fd, total_size)
    try:
        header = [
            SHARED_DEPTH_MAGIC, 1, SHARED_DEPTH_HEADER_SIZE,
            total_size, capacity, capacity,
            width, height, width * 2, 1, 2, 1, 1, 1, 1, 0, 0, 0,
            9, int(time.time() * 1000), os.getpid(), 9, 0,
            100.0, 100.0, 3.5, 2.5,
            *([0] * 12),
        ]
        SHARED_DEPTH_HEADER.pack_into(mapping, 0, *header)
        depth = np.ndarray(
            (height, width),
            dtype="<u2",
            buffer=mapping,
            offset=SHARED_DEPTH_HEADER_SIZE + capacity,
        )
        depth[:] = 1000
        reader = SharedDepthReader(name, max_age_ms=1500)
        try:
            points, metadata = reader.sample_deproject(
                [[4, 3, 4, 3]], width, height, 1, 50.0, 1, 100, 5000
            )
        finally:
            reader.close()
        assert points[0]["depth_valid"] is True
        assert points[0]["depth_mm"] == 1000
        assert points[0]["position_camera"] == [5.0, 5.0, 1000.0]
        assert metadata["mode"] == "shared_depth"
        assert metadata["depth_sequence"] == 9
    finally:
        mapping.close()
        os.close(fd)
        os.unlink(path)


def test_shared_depth_reader_scales_display_coordinates_to_depth_coordinates():
    name = f"/visionops_test_depth_scaled_{os.getpid()}"
    path = "/dev/shm/" + name.lstrip("/")
    width, height = 8, 6
    capacity = width * height * 2
    total_size = SHARED_DEPTH_HEADER_SIZE + 2 * capacity
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    os.ftruncate(fd, total_size)
    mapping = mmap.mmap(fd, total_size)
    try:
        header = [
            SHARED_DEPTH_MAGIC, 1, SHARED_DEPTH_HEADER_SIZE,
            total_size, capacity, capacity,
            width, height, width * 2, 1, 2, 1, 0, 1, 1, 0, 0, 0,
            12, int(time.time() * 1000), os.getpid(), 12, 0,
            100.0, 100.0, 3.5, 2.5,
            *([0] * 12),
        ]
        SHARED_DEPTH_HEADER.pack_into(mapping, 0, *header)
        depth = np.ndarray(
            (height, width),
            dtype="<u2",
            buffer=mapping,
            offset=SHARED_DEPTH_HEADER_SIZE,
        )
        depth[:] = 1000
        reader = SharedDepthReader(name, max_age_ms=1500)
        try:
            points, metadata = reader.sample_deproject(
                [[8, 6, 8, 6]], 16, 12, 1, 50.0, 1, 100, 5000
            )
            reader_status = reader.status()
        finally:
            reader.close()
        assert points[0]["sample_px"] == [4, 3]
        assert points[0]["position_camera"] == [5.0, 5.0, 1000.0]
        assert metadata["depth_sequence"] == 12
        assert reader_status["mapped"] is True
        assert reader_status["mapping_size"] == total_size
    finally:
        mapping.close()
        os.close(fd)
        os.unlink(path)
