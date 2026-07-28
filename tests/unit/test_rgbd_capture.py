"""Synchronized Orbbec shared-memory RGB-D capture tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from apps.collector_web.backend import rgbd_capture as module


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _put_f64(buffer: bytearray, offset: int, value: float) -> None:
    struct.pack_into("<d", buffer, offset, value)


def _write_rgb(path: Path, timestamp: int) -> None:
    width, height = 4, 3
    stride = width * 3
    frame = np.arange(width * height * 3, dtype=np.uint8).tobytes()
    capacity = len(frame)
    total = module.RGB_HEADER_SIZE + capacity * module.RGB_BUFFER_COUNT
    body = bytearray(total)
    _put_u64(body, 0, module.RGB_MAGIC)
    _put_u32(body, 8, module.RGB_VERSION)
    _put_u32(body, 12, module.RGB_HEADER_SIZE)
    _put_u64(body, 16, total)
    _put_u64(body, 24, capacity)
    _put_u64(body, 32, len(frame))
    _put_u32(body, 40, width)
    _put_u32(body, 44, height)
    _put_u32(body, 48, 3)
    _put_u32(body, 52, stride)
    _put_u32(body, 56, module.RGB_PIXEL_FORMAT_RGB888)
    _put_u32(body, 60, module.RGB_BUFFER_COUNT)
    _put_u32(body, 64, module.RGB_STATE_RUNNING)
    _put_u32(body, 68, 1)
    _put_u64(body, 72, 7)
    _put_u64(body, 80, timestamp)
    start = module.RGB_HEADER_SIZE + capacity
    body[start:start + len(frame)] = frame
    path.write_bytes(body)


def _write_depth(path: Path, timestamp: int) -> None:
    width, height = 4, 3
    depth = np.full((height, width), 999, dtype=np.uint16)
    frame = depth.tobytes()
    stride = width * 2
    capacity = len(frame)
    total = module.DEPTH_HEADER_SIZE + capacity * module.DEPTH_BUFFER_COUNT
    body = bytearray(total)
    _put_u64(body, 0, module.DEPTH_MAGIC)
    _put_u32(body, 8, module.DEPTH_VERSION)
    _put_u32(body, 12, module.DEPTH_HEADER_SIZE)
    _put_u64(body, 16, total)
    _put_u64(body, 24, capacity)
    _put_u64(body, 32, len(frame))
    _put_u32(body, 40, width)
    _put_u32(body, 44, height)
    _put_u32(body, 48, stride)
    _put_u32(body, 52, module.DEPTH_PIXEL_FORMAT_UINT16_MM)
    _put_u32(body, 56, module.DEPTH_BUFFER_COUNT)
    _put_u32(body, 60, module.DEPTH_STATE_RUNNING)
    _put_u32(body, 64, 1)
    _put_u32(body, 68, 1)
    _put_u32(body, 72, 1)
    _put_u64(body, 88, 9)
    _put_u64(body, 96, timestamp)
    _put_f64(body, 128, 200.0)
    _put_f64(body, 136, 201.0)
    _put_f64(body, 144, 2.0)
    _put_f64(body, 152, 1.5)
    start = module.DEPTH_HEADER_SIZE + capacity
    body[start:start + len(frame)] = frame
    path.write_bytes(body)


def test_capture_accepts_only_timestamp_matched_shared_frames(monkeypatch, tmp_path: Path) -> None:
    timestamp = module.time.time_ns() // 1_000_000
    rgb_path = tmp_path / "rgb.shm"
    depth_path = tmp_path / "depth.shm"
    _write_rgb(rgb_path, timestamp)
    _write_depth(depth_path, timestamp)
    monkeypatch.setenv("VISIONOPS_CAPTURE_SHARED_RGB_PATH", str(rgb_path))
    monkeypatch.setenv("VISIONOPS_CAPTURE_SHARED_DEPTH_PATH", str(depth_path))
    monkeypatch.setattr(
        module,
        "active_camera_spec",
        lambda: {
            "camera_model": "orbbec336l",
            "display_name": "Orbbec",
            "base_url": "http://127.0.0.1:18182",
            "selection_path": "/tmp/active_camera.json",
        },
    )

    result = module.capture_synchronized_rgbd(timeout_seconds=0.2)
    assert result["synchronized"] is True
    assert result["timestamp_epoch_ms"] == timestamp
    assert result["rgb"].sequence == 7
    assert result["depth"].sequence == 9
    assert result["depth"].fx == 200.0
