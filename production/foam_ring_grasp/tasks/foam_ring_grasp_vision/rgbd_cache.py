"""Persistent synchronized RGB-D shared-memory cache for foam-ring online vision.

M36.3 keeps a short history of exact RGB/depth pairs published by the Orbbec
336L Bridge.  Runtime returns the RGB capture timestamp after RKNN inference;
callers use :meth:`RgbdFrameCache.get_exact` to retrieve the depth frame from
that same SDK FrameSet.  The cache never silently substitutes a neighbouring or
latest depth frame.

The Bridge publishes RGB and D2C-aligned uint16-millimetre depth into separate
double-buffered POSIX shared-memory objects.  Both headers receive the same
``timestamp_epoch_ms`` inside one FrameSet callback.  This module validates the
sequence counter before and after each copy and additionally checks that the
active buffer corresponds to the published sequence, preventing a reader from
accepting metadata that is in the middle of being published.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

RGB_MAGIC = 0x564F505352474231  # VOPSRGB1
RGB_VERSION = 1
RGB_HEADER_SIZE = 192
RGB_STATE_RUNNING = 1
RGB_PIXEL_FORMAT_RGB888 = 1
RGB_BUFFER_COUNT = 2

DEPTH_MAGIC = 0x564F505344455031  # VOPSDEP1
DEPTH_VERSION = 1
DEPTH_HEADER_SIZE = 256
DEPTH_STATE_RUNNING = 1
DEPTH_PIXEL_FORMAT_UINT16_MM = 1
DEPTH_BUFFER_COUNT = 2


class SharedRgbdUnavailable(RuntimeError):
    """The requested synchronized shared-memory data is unavailable."""


class SharedMemoryConcurrentUpdate(SharedRgbdUnavailable):
    """The Bridge published a new buffer while the reader was copying."""




@dataclass(frozen=True)
class RgbdCacheSettings:
    enabled: bool = True
    rgb_name: str = "/visionops_orbbec336l_rgb"
    depth_name: str = "/visionops_orbbec336l_depth"
    cache_frames: int = 12
    max_age_ms: int = 2000
    poll_interval_ms: float = 1.0
    exact_match_timeout_ms: int = 500
    cache_rgb: bool = True
    require_exact_timestamp: bool = True
    allow_nearest_fallback: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RgbdCacheSettings":
        data = dict(value or {})
        settings = cls(
            enabled=bool(data.get("enabled", True)),
            rgb_name=str(data.get("shared_rgb_name") or cls.rgb_name),
            depth_name=str(data.get("shared_depth_name") or cls.depth_name),
            cache_frames=max(2, int(data.get("cache_frames", cls.cache_frames))),
            max_age_ms=max(100, int(data.get("max_age_ms", cls.max_age_ms))),
            poll_interval_ms=max(0.5, float(data.get("poll_interval_ms", cls.poll_interval_ms))),
            exact_match_timeout_ms=max(0, int(data.get("exact_match_timeout_ms", cls.exact_match_timeout_ms))),
            cache_rgb=bool(data.get("cache_rgb", True)),
            require_exact_timestamp=bool(data.get("require_exact_timestamp", True)),
            allow_nearest_fallback=bool(data.get("allow_nearest_fallback", False)),
        )
        if not settings.require_exact_timestamp or settings.allow_nearest_fallback:
            raise ValueError(
                "M36.3 requires exact timestamp matching and forbids nearest-frame fallback"
            )
        return settings


def load_rgbd_cache_settings(path: str | Path) -> RgbdCacheSettings:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件必须是YAML对象: {config_path}")
    section = raw.get("online_rgbd") or {}
    if not isinstance(section, dict):
        raise ValueError("line.yaml 中 online_rgbd 必须是对象")
    return RgbdCacheSettings.from_mapping(section)


@dataclass(frozen=True)
class SharedFrameMetadata:
    sequence: int
    timestamp_epoch_ms: int
    width: int
    height: int
    stride_bytes: int
    active_buffer: int
    state: int


@dataclass(frozen=True)
class SharedRgbFrame:
    rgb: np.ndarray
    sequence: int
    timestamp_epoch_ms: int
    width: int
    height: int
    stride_bytes: int


@dataclass(frozen=True)
class SharedDepthFrame:
    depth_mm: np.ndarray
    sequence: int
    timestamp_epoch_ms: int
    width: int
    height: int
    stride_bytes: int
    aligned_to_color: bool
    calibration_ready: bool
    flip_horizontal: bool
    flip_vertical: bool
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbdFrame:
    """One exact RGB-D pair copied from a single Orbbec SDK FrameSet."""

    timestamp_epoch_ms: int
    rgb_sequence: int
    depth_sequence: int
    rgb: np.ndarray | None
    depth_mm: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    aligned_to_color: bool
    calibration_ready: bool
    flip_horizontal: bool
    flip_vertical: bool
    cached_monotonic: float

    def metadata(self) -> dict[str, Any]:
        return {
            "timestamp_epoch_ms": int(self.timestamp_epoch_ms),
            "rgb_sequence": int(self.rgb_sequence),
            "depth_sequence": int(self.depth_sequence),
            "width": int(self.width),
            "height": int(self.height),
            "intrinsics": {
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
            },
            "aligned_to_color": bool(self.aligned_to_color),
            "calibration_ready": bool(self.calibration_ready),
            "flip_horizontal": bool(self.flip_horizontal),
            "flip_vertical": bool(self.flip_vertical),
            "rgb_cached": self.rgb is not None,
        }


def _u32(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", buffer, offset)[0]


def _u64(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", buffer, offset)[0]


def _f64(buffer: mmap.mmap, offset: int) -> float:
    return struct.unpack_from("<d", buffer, offset)[0]


def shared_memory_path(name_or_path: str) -> Path:
    value = str(name_or_path or "").strip()
    if not value:
        raise SharedRgbdUnavailable("共享内存名称为空")
    candidate = Path(value).expanduser()
    # A path below a real directory is useful for unit tests and custom mounts.
    if candidate.is_absolute() and candidate.parent != Path("/"):
        return candidate
    return Path("/dev/shm") / value.lstrip("/")


class _PersistentMapping:
    """Read-only mmap that automatically reopens after Bridge recreation."""

    def __init__(self, name_or_path: str, minimum_size: int) -> None:
        self.path = shared_memory_path(name_or_path)
        self.minimum_size = int(minimum_size)
        self._fd = -1
        self._mapping: mmap.mmap | None = None
        self._identity: tuple[int, int, int] | None = None
        self.reopen_count = 0
        self.last_error: str | None = None
        self._last_stat_check_monotonic = 0.0
        self._stat_check_interval_s = 0.25

    @property
    def mapping(self) -> mmap.mmap:
        self.ensure_open()
        if self._mapping is None:  # pragma: no cover - defensive guard
            raise SharedRgbdUnavailable(f"共享内存未映射: {self.path}")
        return self._mapping

    def ensure_open(self) -> None:
        now = time.monotonic()
        if (
            self._mapping is not None
            and now - self._last_stat_check_monotonic < self._stat_check_interval_s
        ):
            return
        try:
            current = self.path.stat()
        except OSError as error:
            self.close()
            self.last_error = f"无法访问共享内存 {self.path}: {error}"
            raise SharedRgbdUnavailable(self.last_error) from error
        self._last_stat_check_monotonic = now
        identity = (int(current.st_dev), int(current.st_ino), int(current.st_size))
        if current.st_size < self.minimum_size:
            self.close()
            self.last_error = f"共享内存 {self.path} 大小异常: {current.st_size}"
            raise SharedRgbdUnavailable(self.last_error)
        if self._mapping is not None and self._identity == identity:
            return
        self.close()
        try:
            self._fd = os.open(self.path, os.O_RDONLY)
            mapped_size = os.fstat(self._fd).st_size
            if mapped_size < self.minimum_size:
                raise SharedRgbdUnavailable(f"共享内存 {self.path} 大小异常: {mapped_size}")
            self._mapping = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
            stat = os.fstat(self._fd)
            self._identity = (int(stat.st_dev), int(stat.st_ino), int(stat.st_size))
            self.reopen_count += 1
            self.last_error = None
        except Exception as error:
            self.close()
            self.last_error = f"无法打开共享内存 {self.path}: {error}"
            if isinstance(error, SharedRgbdUnavailable):
                raise
            raise SharedRgbdUnavailable(self.last_error) from error

    def close(self) -> None:
        if self._mapping is not None:
            try:
                self._mapping.close()
            except (BufferError, OSError):
                pass
        self._mapping = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = -1
        self._identity = None
        self._last_stat_check_monotonic = 0.0


class SharedRgbReader:
    def __init__(self, name_or_path: str = "/visionops_orbbec336l_rgb") -> None:
        self._source = _PersistentMapping(name_or_path, RGB_HEADER_SIZE)

    @property
    def path(self) -> Path:
        return self._source.path

    def close(self) -> None:
        self._source.close()

    def peek(self) -> SharedFrameMetadata:
        mapping = self._source.mapping
        if _u64(mapping, 0) != RGB_MAGIC or _u32(mapping, 8) != RGB_VERSION:
            raise SharedRgbdUnavailable(f"RGB共享内存协议不匹配: {self.path}")
        sequence = _u64(mapping, 72)
        return SharedFrameMetadata(
            sequence=sequence,
            timestamp_epoch_ms=_u64(mapping, 80),
            width=_u32(mapping, 40),
            height=_u32(mapping, 44),
            stride_bytes=_u32(mapping, 52),
            active_buffer=_u32(mapping, 68),
            state=_u32(mapping, 64),
        )

    def read_latest(self) -> SharedRgbFrame:
        mapping = self._source.mapping
        if _u64(mapping, 0) != RGB_MAGIC or _u32(mapping, 8) != RGB_VERSION:
            raise SharedRgbdUnavailable(f"RGB共享内存协议不匹配: {self.path}")
        header_size = _u32(mapping, 12)
        total_size = _u64(mapping, 16)
        frame_capacity = _u64(mapping, 24)
        frame_bytes = _u64(mapping, 32)
        width = _u32(mapping, 40)
        height = _u32(mapping, 44)
        channels = _u32(mapping, 48)
        stride = _u32(mapping, 52)
        pixel_format = _u32(mapping, 56)
        buffer_count = _u32(mapping, 60)
        state = _u32(mapping, 64)
        sequence_before = _u64(mapping, 72)
        active = _u32(mapping, 68)
        timestamp = _u64(mapping, 80)
        if (
            header_size != RGB_HEADER_SIZE
            or total_size > len(mapping)
            or channels != 3
            or pixel_format != RGB_PIXEL_FORMAT_RGB888
            or buffer_count != RGB_BUFFER_COUNT
            or state != RGB_STATE_RUNNING
            or sequence_before <= 0
            or active != sequence_before % buffer_count
            or width <= 0
            or height <= 0
            or stride < width * 3
            or frame_bytes < stride * height
            or frame_bytes > frame_capacity
        ):
            raise SharedRgbdUnavailable("RGB共享内存尚未就绪或元数据无效")
        start = header_size + frame_capacity * active
        end = start + frame_bytes
        if end > len(mapping):
            raise SharedRgbdUnavailable("RGB共享内存帧越界")
        view = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=mapping,
            offset=start,
            strides=(stride, 3, 1),
        )
        rgb = view.copy(order="C")
        sequence_after = _u64(mapping, 72)
        active_after = _u32(mapping, 68)
        timestamp_after = _u64(mapping, 80)
        if (
            sequence_before != sequence_after
            or active != active_after
            or timestamp != timestamp_after
            or active_after != sequence_after % buffer_count
        ):
            raise SharedMemoryConcurrentUpdate("读取RGB共享帧时发生并发更新")
        rgb.setflags(write=False)
        return SharedRgbFrame(
            rgb=rgb,
            sequence=sequence_after,
            timestamp_epoch_ms=timestamp_after,
            width=width,
            height=height,
            stride_bytes=stride,
        )

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "mapped": self._source._mapping is not None,
            "reopen_count": int(self._source.reopen_count),
            "last_error": self._source.last_error,
        }


class SharedDepthReader:
    def __init__(self, name_or_path: str = "/visionops_orbbec336l_depth") -> None:
        self._source = _PersistentMapping(name_or_path, DEPTH_HEADER_SIZE)

    @property
    def path(self) -> Path:
        return self._source.path

    def close(self) -> None:
        self._source.close()

    def peek(self) -> SharedFrameMetadata:
        mapping = self._source.mapping
        if _u64(mapping, 0) != DEPTH_MAGIC or _u32(mapping, 8) != DEPTH_VERSION:
            raise SharedRgbdUnavailable(f"Depth共享内存协议不匹配: {self.path}")
        return SharedFrameMetadata(
            sequence=_u64(mapping, 88),
            timestamp_epoch_ms=_u64(mapping, 96),
            width=_u32(mapping, 40),
            height=_u32(mapping, 44),
            stride_bytes=_u32(mapping, 48),
            active_buffer=_u32(mapping, 64),
            state=_u32(mapping, 60),
        )

    def read_latest(self) -> SharedDepthFrame:
        mapping = self._source.mapping
        if _u64(mapping, 0) != DEPTH_MAGIC or _u32(mapping, 8) != DEPTH_VERSION:
            raise SharedRgbdUnavailable(f"Depth共享内存协议不匹配: {self.path}")
        header_size = _u32(mapping, 12)
        total_size = _u64(mapping, 16)
        frame_capacity = _u64(mapping, 24)
        frame_bytes = _u64(mapping, 32)
        width = _u32(mapping, 40)
        height = _u32(mapping, 44)
        stride = _u32(mapping, 48)
        pixel_format = _u32(mapping, 52)
        buffer_count = _u32(mapping, 56)
        state = _u32(mapping, 60)
        active = _u32(mapping, 64)
        calibration_ready = _u32(mapping, 68) != 0
        aligned_to_color = _u32(mapping, 72) != 0
        flip_horizontal = _u32(mapping, 76) != 0
        flip_vertical = _u32(mapping, 80) != 0
        sequence_before = _u64(mapping, 88)
        timestamp = _u64(mapping, 96)
        if (
            header_size != DEPTH_HEADER_SIZE
            or total_size > len(mapping)
            or pixel_format != DEPTH_PIXEL_FORMAT_UINT16_MM
            or buffer_count != DEPTH_BUFFER_COUNT
            or state != DEPTH_STATE_RUNNING
            or not calibration_ready
            or not aligned_to_color
            or sequence_before <= 0
            or active != sequence_before % buffer_count
            or width <= 0
            or height <= 0
            or stride < width * 2
            or frame_bytes < stride * height
            or frame_bytes > frame_capacity
        ):
            raise SharedRgbdUnavailable("Depth共享内存尚未就绪、未对齐、未标定或元数据无效")
        start = header_size + frame_capacity * active
        end = start + frame_bytes
        if end > len(mapping):
            raise SharedRgbdUnavailable("Depth共享内存帧越界")
        view = np.ndarray(
            (height, width),
            dtype="<u2",
            buffer=mapping,
            offset=start,
            strides=(stride, 2),
        )
        depth = view.copy(order="C")
        sequence_after = _u64(mapping, 88)
        active_after = _u32(mapping, 64)
        timestamp_after = _u64(mapping, 96)
        if (
            sequence_before != sequence_after
            or active != active_after
            or timestamp != timestamp_after
            or active_after != sequence_after % buffer_count
        ):
            raise SharedMemoryConcurrentUpdate("读取Depth共享帧时发生并发更新")
        depth.setflags(write=False)
        return SharedDepthFrame(
            depth_mm=depth,
            sequence=sequence_after,
            timestamp_epoch_ms=timestamp_after,
            width=width,
            height=height,
            stride_bytes=stride,
            aligned_to_color=aligned_to_color,
            calibration_ready=calibration_ready,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            fx=_f64(mapping, 128),
            fy=_f64(mapping, 136),
            cx=_f64(mapping, 144),
            cy=_f64(mapping, 152),
        )

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "mapped": self._source._mapping is not None,
            "reopen_count": int(self._source.reopen_count),
            "last_error": self._source.last_error,
        }


class RgbdFrameCache:
    """Background cache of exact timestamp-matched RGB-D frames."""

    def __init__(
        self,
        *,
        rgb_name: str = "/visionops_orbbec336l_rgb",
        depth_name: str = "/visionops_orbbec336l_depth",
        max_frames: int = 12,
        max_age_ms: int = 2000,
        poll_interval_ms: float = 1.0,
        cache_rgb: bool = True,
    ) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        self.rgb_reader = SharedRgbReader(rgb_name)
        self.depth_reader = SharedDepthReader(depth_name)
        self.max_frames = int(max_frames)
        self.max_age_ms = max(100, int(max_age_ms))
        self.poll_interval_s = max(0.0005, float(poll_interval_ms) / 1000.0)
        self.cache_rgb = bool(cache_rgb)

        self._frames: OrderedDict[int, RgbdFrame] = OrderedDict()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._insert_times: deque[float] = deque(maxlen=120)

        self._last_cached_timestamp = 0
        self._last_rgb_sequence = 0
        self._last_depth_sequence = 0
        self._last_error: str | None = None
        self._started_monotonic = 0.0
        self._pairs_cached = 0
        self._timestamp_mismatch_count = 0
        self._concurrent_update_count = 0
        self._read_error_count = 0
        self._eviction_count = 0
        self._exact_match_count = 0
        self._exact_miss_count = 0
        self._wait_timeout_count = 0

    def __enter__(self) -> "RgbdFrameCache":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._started_monotonic = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="foam-ring-rgbd-cache",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))
        self._thread = None
        self.rgb_reader.close()
        self.depth_reader.close()

    def wait_until_ready(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._frames and self.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return bool(self._frames)

    def get_exact(self, timestamp_epoch_ms: int, timeout: float = 0.0) -> RgbdFrame | None:
        """Return only the exact frame, never a neighbouring depth frame."""

        timestamp = int(timestamp_epoch_ms)
        if timestamp <= 0:
            with self._condition:
                self._exact_miss_count += 1
            return None
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                frame = self._frames.get(timestamp)
                if frame is not None:
                    self._exact_match_count += 1
                    return frame
                # Once the cache has moved beyond a missing timestamp and its
                # oldest retained frame is newer, waiting cannot recover it.
                if self._frames:
                    oldest = next(iter(self._frames))
                    latest = next(reversed(self._frames))
                    if oldest > timestamp or (
                        latest > timestamp and len(self._frames) >= self.max_frames
                    ):
                        self._exact_miss_count += 1
                        return None
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.running:
                    self._exact_miss_count += 1
                    if timeout > 0:
                        self._wait_timeout_count += 1
                    return None
                self._condition.wait(remaining)

    def latest(self) -> RgbdFrame | None:
        with self._condition:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))

    def timestamps(self) -> list[int]:
        with self._condition:
            return list(self._frames.keys())

    def status(self) -> dict[str, Any]:
        with self._condition:
            timestamps = list(self._frames.keys())
            now_ms = int(time.time() * 1000)
            latest_age_ms = now_ms - timestamps[-1] if timestamps else None
            fps = 0.0
            if len(self._insert_times) >= 2:
                elapsed = self._insert_times[-1] - self._insert_times[0]
                if elapsed > 0:
                    fps = (len(self._insert_times) - 1) / elapsed
            return {
                "running": self.running,
                "cache_rgb": self.cache_rgb,
                "cache_size": len(self._frames),
                "cache_capacity": self.max_frames,
                "max_age_ms": self.max_age_ms,
                "oldest_timestamp_ms": timestamps[0] if timestamps else 0,
                "latest_timestamp_ms": timestamps[-1] if timestamps else 0,
                "latest_age_ms": latest_age_ms,
                "latest_rgb_sequence": self._last_rgb_sequence,
                "latest_depth_sequence": self._last_depth_sequence,
                "pair_fps": round(fps, 3),
                "pairs_cached": self._pairs_cached,
                "timestamp_mismatch_count": self._timestamp_mismatch_count,
                "concurrent_update_count": self._concurrent_update_count,
                "read_error_count": self._read_error_count,
                "eviction_count": self._eviction_count,
                "exact_match_count": self._exact_match_count,
                "exact_miss_count": self._exact_miss_count,
                "wait_timeout_count": self._wait_timeout_count,
                "last_error": self._last_error,
                "rgb_reader": self.rgb_reader.status(),
                "depth_reader": self.depth_reader.status(),
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rgb_meta = self.rgb_reader.peek()
                depth_meta = self.depth_reader.peek()
                if rgb_meta.sequence <= 0 or depth_meta.sequence <= 0:
                    raise SharedRgbdUnavailable("RGB或Depth共享内存尚未发布有效帧")
                if rgb_meta.timestamp_epoch_ms != depth_meta.timestamp_epoch_ms:
                    self._timestamp_mismatch_count += 1
                    self._last_error = (
                        "RGB/Depth时间戳等待匹配: "
                        f"rgb={rgb_meta.timestamp_epoch_ms}, depth={depth_meta.timestamp_epoch_ms}"
                    )
                    self._stop_event.wait(self.poll_interval_s)
                    continue
                if rgb_meta.timestamp_epoch_ms == self._last_cached_timestamp:
                    self._stop_event.wait(self.poll_interval_s)
                    continue

                rgb_frame = self.rgb_reader.read_latest()
                depth_frame = self.depth_reader.read_latest()
                if rgb_frame.timestamp_epoch_ms != depth_frame.timestamp_epoch_ms:
                    self._timestamp_mismatch_count += 1
                    self._last_error = (
                        "复制后RGB/Depth时间戳不匹配: "
                        f"rgb={rgb_frame.timestamp_epoch_ms}, depth={depth_frame.timestamp_epoch_ms}"
                    )
                    self._stop_event.wait(self.poll_interval_s)
                    continue
                if rgb_frame.width != depth_frame.width or rgb_frame.height != depth_frame.height:
                    raise SharedRgbdUnavailable(
                        "RGB/Depth分辨率不一致: "
                        f"rgb={rgb_frame.width}x{rgb_frame.height}, "
                        f"depth={depth_frame.width}x{depth_frame.height}"
                    )
                if depth_frame.fx <= 0 or depth_frame.fy <= 0:
                    raise SharedRgbdUnavailable("Depth共享内存内参无效")

                frame = RgbdFrame(
                    timestamp_epoch_ms=rgb_frame.timestamp_epoch_ms,
                    rgb_sequence=rgb_frame.sequence,
                    depth_sequence=depth_frame.sequence,
                    rgb=rgb_frame.rgb if self.cache_rgb else None,
                    depth_mm=depth_frame.depth_mm,
                    width=rgb_frame.width,
                    height=rgb_frame.height,
                    fx=depth_frame.fx,
                    fy=depth_frame.fy,
                    cx=depth_frame.cx,
                    cy=depth_frame.cy,
                    aligned_to_color=depth_frame.aligned_to_color,
                    calibration_ready=depth_frame.calibration_ready,
                    flip_horizontal=depth_frame.flip_horizontal,
                    flip_vertical=depth_frame.flip_vertical,
                    cached_monotonic=time.monotonic(),
                )
                self._insert(frame)
                self._last_cached_timestamp = frame.timestamp_epoch_ms
                self._last_rgb_sequence = frame.rgb_sequence
                self._last_depth_sequence = frame.depth_sequence
                self._last_error = None
            except SharedMemoryConcurrentUpdate as error:
                self._concurrent_update_count += 1
                self._last_error = str(error)
                self._stop_event.wait(self.poll_interval_s)
            except SharedRgbdUnavailable as error:
                self._read_error_count += 1
                self._last_error = str(error)
                self._stop_event.wait(max(0.01, self.poll_interval_s))
            except Exception as error:  # pragma: no cover - defensive safety net
                self._read_error_count += 1
                self._last_error = f"RGB-D缓存线程异常: {error}"
                self._stop_event.wait(0.05)

    def _insert(self, frame: RgbdFrame) -> None:
        with self._condition:
            self._frames[frame.timestamp_epoch_ms] = frame
            self._frames.move_to_end(frame.timestamp_epoch_ms)
            self._pairs_cached += 1
            self._insert_times.append(frame.cached_monotonic)
            now_ms = int(time.time() * 1000)
            while self._frames:
                oldest_timestamp = next(iter(self._frames))
                too_many = len(self._frames) > self.max_frames
                too_old = now_ms - oldest_timestamp > self.max_age_ms
                if not too_many and not too_old:
                    break
                self._frames.popitem(last=False)
                self._eviction_count += 1
            self._condition.notify_all()
