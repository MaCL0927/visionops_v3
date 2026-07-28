"""Synchronized RGB-D capture from the active Orbbec bridge shared memory.

The Orbbec bridge publishes RGB and D2C-aligned uint16 depth from one SDK
``FrameSet`` with the same ``timestamp_epoch_ms``.  This module copies both
buffers with sequence-counter validation and only accepts a pair whose
published timestamps are identical, so a saved RGB/depth pair cannot silently
mix adjacent camera frames.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge.camera_bridge.camera_selection import active_camera_spec

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


class RgbdCaptureUnavailable(RuntimeError):
    """Exact synchronized RGB-D data is not currently available."""


@dataclass(frozen=True)
class SharedRgbFrame:
    data: bytes
    width: int
    height: int
    stride_bytes: int
    sequence: int
    timestamp_epoch_ms: int


@dataclass(frozen=True)
class SharedDepthFrame:
    data: bytes
    width: int
    height: int
    stride_bytes: int
    sequence: int
    timestamp_epoch_ms: int
    aligned_to_color: bool
    calibration_ready: bool
    flip_horizontal: bool
    flip_vertical: bool
    fx: float
    fy: float
    cx: float
    cy: float


def _u32(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", buffer, offset)[0]


def _u64(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", buffer, offset)[0]


def _f64(buffer: mmap.mmap, offset: int) -> float:
    return struct.unpack_from("<d", buffer, offset)[0]


def _shm_path(name: str) -> Path:
    value = str(name or "").strip()
    if not value:
        raise RgbdCaptureUnavailable("共享内存名称为空")
    # Tests and custom deployments may provide a real filesystem path.
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.parent != Path("/"):
        return candidate
    return Path("/dev/shm") / value.lstrip("/")


def _rgb_shm_path() -> Path:
    override = os.environ.get("VISIONOPS_CAPTURE_SHARED_RGB_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    name = os.environ.get(
        "VISIONOPS_CAPTURE_SHARED_RGB_NAME",
        os.environ.get("VISIONOPS_ORBBEC336L_SHARED_RGB_NAME", "/visionops_orbbec336l_rgb"),
    )
    return _shm_path(name)


def _depth_shm_path() -> Path:
    override = os.environ.get("VISIONOPS_CAPTURE_SHARED_DEPTH_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    name = os.environ.get(
        "VISIONOPS_CAPTURE_SHARED_DEPTH_NAME",
        os.environ.get("VISIONOPS_ORBBEC336L_SHARED_DEPTH_NAME", "/visionops_orbbec336l_depth"),
    )
    return _shm_path(name)


def _open_mapping(path: Path, minimum_size: int) -> tuple[object, mmap.mmap]:
    try:
        handle = path.open("rb", buffering=0)
    except OSError as error:
        raise RgbdCaptureUnavailable(f"无法打开共享内存 {path}: {error}") from error
    try:
        size = os.fstat(handle.fileno()).st_size
        if size < minimum_size:
            raise RgbdCaptureUnavailable(f"共享内存 {path} 大小异常: {size}")
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return handle, mapping
    except Exception:
        handle.close()
        raise


def _read_rgb_once(path: Path) -> SharedRgbFrame:
    handle, mapping = _open_mapping(path, RGB_HEADER_SIZE)
    try:
        if _u64(mapping, 0) != RGB_MAGIC or _u32(mapping, 8) != RGB_VERSION:
            raise RgbdCaptureUnavailable(f"RGB共享内存协议不匹配: {path}")
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
            or width <= 0
            or height <= 0
            or stride < width * 3
            or frame_bytes <= 0
            or frame_bytes > frame_capacity
        ):
            raise RgbdCaptureUnavailable("RGB共享内存尚未就绪或元数据无效")
        start = header_size + frame_capacity * (active % buffer_count)
        end = start + frame_bytes
        if end > len(mapping):
            raise RgbdCaptureUnavailable("RGB共享内存帧越界")
        data = bytes(mapping[start:end])
        sequence_after = _u64(mapping, 72)
        if sequence_before != sequence_after:
            raise RgbdCaptureUnavailable("读取RGB共享帧时发生并发更新")
        return SharedRgbFrame(
            data=data,
            width=width,
            height=height,
            stride_bytes=stride,
            sequence=sequence_after,
            timestamp_epoch_ms=timestamp,
        )
    finally:
        mapping.close()
        handle.close()


def _read_depth_once(path: Path) -> SharedDepthFrame:
    handle, mapping = _open_mapping(path, DEPTH_HEADER_SIZE)
    try:
        if _u64(mapping, 0) != DEPTH_MAGIC or _u32(mapping, 8) != DEPTH_VERSION:
            raise RgbdCaptureUnavailable(f"Depth共享内存协议不匹配: {path}")
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
            or sequence_before <= 0
            or width <= 0
            or height <= 0
            or stride < width * 2
            or frame_bytes <= 0
            or frame_bytes > frame_capacity
        ):
            raise RgbdCaptureUnavailable("Depth共享内存尚未就绪、未标定或元数据无效")
        start = header_size + frame_capacity * (active % buffer_count)
        end = start + frame_bytes
        if end > len(mapping):
            raise RgbdCaptureUnavailable("Depth共享内存帧越界")
        data = bytes(mapping[start:end])
        sequence_after = _u64(mapping, 88)
        if sequence_before != sequence_after:
            raise RgbdCaptureUnavailable("读取Depth共享帧时发生并发更新")
        return SharedDepthFrame(
            data=data,
            width=width,
            height=height,
            stride_bytes=stride,
            sequence=sequence_after,
            timestamp_epoch_ms=timestamp,
            aligned_to_color=aligned_to_color,
            calibration_ready=calibration_ready,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            fx=_f64(mapping, 128),
            fy=_f64(mapping, 136),
            cx=_f64(mapping, 144),
            cy=_f64(mapping, 152),
        )
    finally:
        mapping.close()
        handle.close()


def capture_synchronized_rgbd(timeout_seconds: float = 2.0, max_age_ms: int = 1500) -> dict[str, Any]:
    """Return one exact RGB/depth pair from the active Orbbec bridge.

    The two shared-memory publishers use one timestamp generated inside the same
    SDK FrameSet callback.  A pair is accepted only when both timestamps match.
    """

    spec = active_camera_spec()
    camera_model = str(spec.get("camera_model") or "")
    if camera_model != "orbbec336l":
        raise RgbdCaptureUnavailable(
            f"当前相机 {camera_model or 'unknown'} 尚未提供可验证的同步共享RGB-D；"
            "请切换到 Orbbec 336L/Femto Bridge 或关闭‘同步保存深度’"
        )

    rgb_path = _rgb_shm_path()
    depth_path = _depth_shm_path()
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    last_error = "未读取到匹配帧"
    while time.monotonic() < deadline:
        try:
            rgb = _read_rgb_once(rgb_path)
            depth = _read_depth_once(depth_path)
            now_ms = int(time.time() * 1000)
            rgb_age = now_ms - int(rgb.timestamp_epoch_ms)
            depth_age = now_ms - int(depth.timestamp_epoch_ms)
            if rgb.timestamp_epoch_ms != depth.timestamp_epoch_ms:
                last_error = (
                    f"RGB/Depth时间戳尚未匹配: rgb={rgb.timestamp_epoch_ms}, "
                    f"depth={depth.timestamp_epoch_ms}"
                )
                time.sleep(0.01)
                continue
            if rgb_age < 0 or depth_age < 0 or rgb_age > max_age_ms or depth_age > max_age_ms:
                last_error = f"RGB-D共享帧过期: rgb_age={rgb_age}ms, depth_age={depth_age}ms"
                time.sleep(0.02)
                continue
            return {
                "camera": spec,
                "synchronized": True,
                "synchronization_mode": "posix_shared_memory_timestamp_match",
                "timestamp_epoch_ms": int(rgb.timestamp_epoch_ms),
                "rgb": rgb,
                "depth": depth,
                "rgb_shm_path": str(rgb_path),
                "depth_shm_path": str(depth_path),
            }
        except RgbdCaptureUnavailable as error:
            last_error = str(error)
            time.sleep(0.02)
    raise RgbdCaptureUnavailable(f"无法获得同步RGB-D帧: {last_error}")
