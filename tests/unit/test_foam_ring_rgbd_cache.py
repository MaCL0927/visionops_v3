"""M36.3 exact shared RGB-D cache tests."""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision import rgbd_cache as module


def _u32(buffer: mmap.mmap, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _u64(buffer: mmap.mmap, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _f64(buffer: mmap.mmap, offset: int, value: float) -> None:
    struct.pack_into("<d", buffer, offset, value)


class _Publisher:
    def __init__(self, root: Path, *, width: int = 32, height: int = 24) -> None:
        self.rgb_path = root / "rgb.shm"
        self.depth_path = root / "depth.shm"
        self.width = width
        self.height = height
        self.rgb_capacity = width * height * 3
        self.depth_capacity = width * height * 2
        self.rgb_size = module.RGB_HEADER_SIZE + self.rgb_capacity * 2
        self.depth_size = module.DEPTH_HEADER_SIZE + self.depth_capacity * 2
        self.rgb_fd = os.open(self.rgb_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        self.depth_fd = os.open(self.depth_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(self.rgb_fd, self.rgb_size)
        os.ftruncate(self.depth_fd, self.depth_size)
        self.rgb = mmap.mmap(self.rgb_fd, self.rgb_size)
        self.depth = mmap.mmap(self.depth_fd, self.depth_size)
        self.rgb_sequence = 0
        self.depth_sequence = 0
        self._init_headers()

    def _init_headers(self) -> None:
        _u64(self.rgb, 0, module.RGB_MAGIC)
        _u32(self.rgb, 8, module.RGB_VERSION)
        _u32(self.rgb, 12, module.RGB_HEADER_SIZE)
        _u64(self.rgb, 16, self.rgb_size)
        _u64(self.rgb, 24, self.rgb_capacity)
        _u64(self.rgb, 32, self.rgb_capacity)
        _u32(self.rgb, 40, self.width)
        _u32(self.rgb, 44, self.height)
        _u32(self.rgb, 48, 3)
        _u32(self.rgb, 52, self.width * 3)
        _u32(self.rgb, 56, module.RGB_PIXEL_FORMAT_RGB888)
        _u32(self.rgb, 60, module.RGB_BUFFER_COUNT)
        _u32(self.rgb, 64, module.RGB_STATE_RUNNING)

        _u64(self.depth, 0, module.DEPTH_MAGIC)
        _u32(self.depth, 8, module.DEPTH_VERSION)
        _u32(self.depth, 12, module.DEPTH_HEADER_SIZE)
        _u64(self.depth, 16, self.depth_size)
        _u64(self.depth, 24, self.depth_capacity)
        _u64(self.depth, 32, self.depth_capacity)
        _u32(self.depth, 40, self.width)
        _u32(self.depth, 44, self.height)
        _u32(self.depth, 48, self.width * 2)
        _u32(self.depth, 52, module.DEPTH_PIXEL_FORMAT_UINT16_MM)
        _u32(self.depth, 56, module.DEPTH_BUFFER_COUNT)
        _u32(self.depth, 60, module.DEPTH_STATE_RUNNING)
        _u32(self.depth, 68, 1)
        _u32(self.depth, 72, 1)
        _f64(self.depth, 128, 500.0)
        _f64(self.depth, 136, 501.0)
        _f64(self.depth, 144, self.width / 2.0)
        _f64(self.depth, 152, self.height / 2.0)

    def publish_rgb(self, timestamp: int, value: int = 7) -> int:
        self.rgb_sequence += 1
        active = self.rgb_sequence % 2
        start = module.RGB_HEADER_SIZE + active * self.rgb_capacity
        frame = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        self.rgb[start:start + self.rgb_capacity] = frame.tobytes()
        _u32(self.rgb, 68, active)
        _u64(self.rgb, 80, timestamp)
        _u64(self.rgb, 96, self.rgb_sequence)
        _u64(self.rgb, 72, self.rgb_sequence)
        return self.rgb_sequence

    def publish_depth(self, timestamp: int, value: int = 1000) -> int:
        self.depth_sequence += 1
        active = self.depth_sequence % 2
        start = module.DEPTH_HEADER_SIZE + active * self.depth_capacity
        frame = np.full((self.height, self.width), value, dtype=np.uint16)
        self.depth[start:start + self.depth_capacity] = frame.tobytes()
        _u32(self.depth, 64, active)
        _u64(self.depth, 96, timestamp)
        _u64(self.depth, 112, self.depth_sequence)
        _u64(self.depth, 88, self.depth_sequence)
        return self.depth_sequence

    def publish_pair(self, timestamp: int, value: int = 1000) -> tuple[int, int]:
        return self.publish_rgb(timestamp), self.publish_depth(timestamp, value)

    def close(self) -> None:
        self.rgb.close()
        self.depth.close()
        os.close(self.rgb_fd)
        os.close(self.depth_fd)


def test_cache_matches_exact_runtime_timestamp_and_copies_aligned_depth(tmp_path: Path) -> None:
    publisher = _Publisher(tmp_path)
    cache = module.RgbdFrameCache(
        rgb_name=str(publisher.rgb_path),
        depth_name=str(publisher.depth_path),
        max_frames=4,
        max_age_ms=5000,
        poll_interval_ms=0.5,
    )
    try:
        cache.start()
        timestamp = int(time.time() * 1000)
        rgb_sequence, depth_sequence = publisher.publish_pair(timestamp, 1234)
        assert cache.wait_until_ready(1.0)
        frame = cache.get_exact(timestamp, timeout=0.5)
        assert frame is not None
        assert frame.timestamp_epoch_ms == timestamp
        assert frame.rgb_sequence == rgb_sequence
        assert frame.depth_sequence == depth_sequence
        assert frame.rgb is not None
        assert frame.rgb.shape == (24, 32, 3)
        assert frame.depth_mm.shape == (24, 32)
        assert int(frame.depth_mm[5, 6]) == 1234
        assert frame.aligned_to_color is True
        assert frame.fx == 500.0
        assert frame.rgb.flags.writeable is False
        assert frame.depth_mm.flags.writeable is False
    finally:
        cache.stop()
        publisher.close()


def test_cache_rejects_mismatched_timestamps_until_exact_pair_arrives(tmp_path: Path) -> None:
    publisher = _Publisher(tmp_path)
    cache = module.RgbdFrameCache(
        rgb_name=str(publisher.rgb_path),
        depth_name=str(publisher.depth_path),
        max_frames=4,
        max_age_ms=5000,
        poll_interval_ms=0.5,
    )
    try:
        cache.start()
        base = int(time.time() * 1000)
        publisher.publish_rgb(base)
        publisher.publish_depth(base - 33)
        time.sleep(0.03)
        assert cache.get_exact(base, timeout=0.0) is None
        publisher.publish_depth(base)
        frame = cache.get_exact(base, timeout=0.5)
        assert frame is not None
        status = cache.status()
        assert status["timestamp_mismatch_count"] > 0
    finally:
        cache.stop()
        publisher.close()


def test_cache_retains_only_configured_history_and_never_returns_nearest(tmp_path: Path) -> None:
    publisher = _Publisher(tmp_path)
    cache = module.RgbdFrameCache(
        rgb_name=str(publisher.rgb_path),
        depth_name=str(publisher.depth_path),
        max_frames=3,
        max_age_ms=5000,
        poll_interval_ms=0.5,
        cache_rgb=False,
    )
    try:
        cache.start()
        base = int(time.time() * 1000)
        timestamps: list[int] = []
        for index in range(5):
            timestamp = base + index
            timestamps.append(timestamp)
            publisher.publish_pair(timestamp, 900 + index)
            deadline = time.monotonic() + 0.5
            while timestamp not in cache.timestamps() and time.monotonic() < deadline:
                time.sleep(0.002)
        assert cache.timestamps() == timestamps[-3:]
        assert cache.get_exact(timestamps[0], timeout=0.0) is None
        latest = cache.get_exact(timestamps[-1], timeout=0.0)
        assert latest is not None
        assert latest.rgb is None
        assert int(latest.depth_mm[0, 0]) == 904
        assert cache.status()["eviction_count"] >= 2
    finally:
        cache.stop()
        publisher.close()


def test_line_config_enforces_exact_matching_contract(tmp_path: Path) -> None:
    config = tmp_path / "line.yaml"
    config.write_text(
        """
online_rgbd:
  enabled: true
  shared_rgb_name: /rgb
  shared_depth_name: /depth
  cache_frames: 8
  exact_match_timeout_ms: 300
  require_exact_timestamp: true
  allow_nearest_fallback: false
""",
        encoding="utf-8",
    )
    settings = module.load_rgbd_cache_settings(config)
    assert settings.rgb_name == "/rgb"
    assert settings.depth_name == "/depth"
    assert settings.cache_frames == 8
    assert settings.exact_match_timeout_ms == 300

    config.write_text(
        """
online_rgbd:
  require_exact_timestamp: false
  allow_nearest_fallback: true
""",
        encoding="utf-8",
    )
    try:
        module.load_rgbd_cache_settings(config)
    except ValueError as error:
        assert "exact timestamp" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsafe nearest-frame fallback must be rejected")


def test_cached_arrays_are_independent_from_double_buffer_reuse(tmp_path: Path) -> None:
    publisher = _Publisher(tmp_path)
    cache = module.RgbdFrameCache(
        rgb_name=str(publisher.rgb_path),
        depth_name=str(publisher.depth_path),
        max_frames=6,
        max_age_ms=5000,
        poll_interval_ms=0.5,
    )
    try:
        cache.start()
        base = int(time.time() * 1000)
        publisher.publish_pair(base, 1111)
        first = cache.get_exact(base, timeout=0.5)
        assert first is not None
        # Two more publications reuse the same shared-memory buffer index as
        # the first frame. The cached NumPy arrays must remain unchanged.
        publisher.publish_pair(base + 1, 2222)
        time.sleep(0.01)
        publisher.publish_pair(base + 2, 3333)
        deadline = time.monotonic() + 0.5
        while base + 2 not in cache.timestamps() and time.monotonic() < deadline:
            time.sleep(0.002)
        assert int(first.depth_mm[0, 0]) == 1111
        assert first.rgb is not None
        assert int(first.rgb[0, 0, 0]) == 7
    finally:
        cache.stop()
        publisher.close()
