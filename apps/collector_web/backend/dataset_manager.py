"""VisionOps v3 数据集采集、打包与上传管理。

采集数据约定：
- RGB 图片保存到 /opt/visionops_v3/data/images。
- 可选同步深度保存到 /opt/visionops_v3/data/depth（16UC1 PNG，单位毫米）。
- RGB-D 元数据保存到 /opt/visionops_v3/data/meta。
- 上传包保存到 /opt/visionops_v3/data/upload_packages。
- Web 列表分页读取，避免一次性加载大量图片造成页面卡顿。
- 上传配置读取 vision_box_settings.json 中的 upload 字段。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import CollectorConfig
from .capture_roi import (
    CAPTURE_ROI_CONFIG_PATH,
    capture_roi_signature,
    load_capture_roi_config,
    normalize_capture_roi,
    resolve_capture_roi_for_image,
    save_capture_roi_config,
)
from .response_utils import timestamp_ms
from .runtime_client import RuntimeClient, RuntimeUnavailable
from .rgbd_capture import RgbdCaptureUnavailable, capture_synchronized_rgbd
from .vision_box_settings import DEFAULT_PROJECT_ROOT, load_vision_box_settings

IMAGE_DIR = Path(os.environ.get("VISIONOPS_DATASET_IMAGE_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "images")))
DEPTH_DIR = Path(os.environ.get("VISIONOPS_DATASET_DEPTH_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "depth")))
META_DIR = Path(os.environ.get("VISIONOPS_DATASET_META_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "meta")))
PACKAGE_DIR = Path(os.environ.get("VISIONOPS_DATASET_PACKAGE_DIR", str(DEFAULT_PROJECT_ROOT / "data" / "upload_packages")))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
PACKAGE_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_LIMIT = 24
MAX_LIMIT = 100
_DATASET_LOCK = threading.RLock()


class CaptureRoiConflict(ValueError):
    """已有采集图片时尝试切换采集 ROI。"""


def _safe_id(value: str, fallback: str) -> str:
    cleaned = PACKAGE_ID_SAFE.sub("-", str(value or "").strip()).strip(".-_")
    return cleaned or fallback


def _created_at_now() -> tuple[str, str]:
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(), now.strftime("%Y%m%d_%H%M%S")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int(time.time() * 1000) % 1000:03d}"


def _ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name != filename or not SAFE_NAME.match(name):
        raise ValueError("非法文件名")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("不支持的图片扩展名")
    return name


def _image_path(filename: str) -> Path:
    return IMAGE_DIR / _safe_filename(filename)


def _package_path(filename: str) -> Path:
    name = Path(filename).name
    if not name or name != filename or not SAFE_NAME.match(name) or not name.endswith(".tar.gz"):
        raise ValueError("非法压缩包文件名")
    return PACKAGE_DIR / name


def _image_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    name = path.name
    depth_path = DEPTH_DIR / f"{path.stem}.png"
    meta_path = META_DIR / f"{path.stem}.json"
    return {
        "id": path.stem,
        "filename": name,
        "url": f"/api/dataset/images/{name}/content",
        "delete_url": f"/api/dataset/images/{name}",
        "mtime_ms": int(stat.st_mtime * 1000),
        "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "size_bytes": stat.st_size,
        "has_depth": depth_path.is_file(),
        "has_meta": meta_path.is_file(),
        "depth_filename": depth_path.name if depth_path.is_file() else "",
        "meta_filename": meta_path.name if meta_path.is_file() else "",
    }


def list_images(offset: int = 0, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    _ensure_dirs()
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))
    files = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(files)
    depth_count = sum(1 for path in files if (DEPTH_DIR / f"{path.stem}.png").is_file())
    selected = files[safe_offset:safe_offset + safe_limit]
    return {
        "schema_version": "1.0",
        "message_type": "dataset_image_list",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "image_dir": str(IMAGE_DIR),
        "offset": safe_offset,
        "limit": safe_limit,
        "total": total,
        "depth_count": depth_count,
        "has_more": safe_offset + safe_limit < total,
        "images": [_image_record(path) for path in selected],
    }


def get_image_file(filename: str) -> tuple[Path, str]:
    path = _image_path(filename)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(filename)
    ext = path.suffix.lower()
    content_type = "image/png" if ext == ".png" else "image/jpeg"
    return path, content_type


def delete_image(filename: str) -> dict[str, Any]:
    with _DATASET_LOCK:
        path = _image_path(filename)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(filename)
        path.unlink()
        deleted_companions: list[str] = []
        for companion in (DEPTH_DIR / f"{path.stem}.png", META_DIR / f"{path.stem}.json"):
            if companion.is_file():
                companion.unlink()
                deleted_companions.append(str(companion))
        return {
            "schema_version": "1.0",
            "message_type": "dataset_image_delete_result",
            "status": "ok",
            "timestamp_ms": timestamp_ms(),
            "deleted": True,
            "filename": path.name,
            "image_dir": str(IMAGE_DIR),
            "deleted_companions": deleted_companions,
        }


def _dataset_images() -> list[Path]:
    _ensure_dirs()
    return [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]


def get_capture_roi_state() -> dict[str, Any]:
    config = load_capture_roi_config()
    images = _dataset_images()
    return {
        "schema_version": "1.0",
        "message_type": "capture_roi_state",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "capture_roi": config,
        "image_count": len(images),
        "batch_locked": bool(images),
        "config_path": str(CAPTURE_ROI_CONFIG_PATH),
    }


def update_capture_roi(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    desired = normalize_capture_roi(raw)
    with _DATASET_LOCK:
        current = load_capture_roi_config()
        images = _dataset_images()
        changed = capture_roi_signature(desired) != capture_roi_signature(current)
        clear_existing = raw.get("clear_existing_images") is True
        if changed and images and not clear_existing:
            raise CaptureRoiConflict(
                f"当前采集目录已有 {len(images)} 张图片。为避免一个数据包混入多套 ROI，"
                "请先清空采集图片，或确认清空后再应用新 ROI。"
            )
        deleted_count = 0
        if changed and images and clear_existing:
            for image in images:
                for companion in (DEPTH_DIR / f"{image.stem}.png", META_DIR / f"{image.stem}.json"):
                    if companion.is_file():
                        companion.unlink()
                image.unlink()
                deleted_count += 1
        saved = save_capture_roi_config(desired)
        remaining = _dataset_images()
        return {
            "schema_version": "1.0",
            "message_type": "capture_roi_update_result",
            "status": "ok",
            "timestamp_ms": timestamp_ms(),
            "capture_roi": saved,
            "image_count": len(remaining),
            "batch_locked": bool(remaining),
            "deleted_image_count": deleted_count,
        }


def _encode_cropped_snapshot(body: bytes, content_type: str, capture_roi: dict[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    # cv2/numpy 由 RK3576 系统 apt 包提供，Collector venv 使用 --system-site-packages。
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    encoded = np.frombuffer(body, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Runtime 快照不是可解码的 JPEG/PNG 图像")
    height, width = image.shape[:2]
    effective_roi = resolve_capture_roi_for_image(capture_roi, width, height)
    x1, y1, x2, y2 = effective_roi["pixel_xyxy"]
    cropped = np.ascontiguousarray(image[y1:y2, x1:x2])
    if cropped.size == 0:
        raise ValueError("采集 ROI 裁剪结果为空")

    use_png = content_type == "image/png"
    extension = ".png" if use_png else ".jpg"
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if use_png else [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, result = cv2.imencode(extension, cropped, params)
    if not ok:
        raise ValueError("采集 ROI 图片编码失败")
    return result.tobytes(), extension, effective_roi


def _encode_shared_rgb(
    frame: Any,
    capture_roi: dict[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Decode RGB888 shared memory, apply capture ROI, and encode JPEG."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    expected = int(frame.stride_bytes) * int(frame.height)
    if len(frame.data) < expected:
        raise ValueError("RGB共享帧数据长度不足")
    rows = np.frombuffer(frame.data, dtype=np.uint8, count=expected).reshape(frame.height, frame.stride_bytes)
    rgb = rows[:, : frame.width * 3].reshape(frame.height, frame.width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    effective_roi = resolve_capture_roi_for_image(capture_roi, frame.width, frame.height)
    x1, y1, x2, y2 = effective_roi["pixel_xyxy"]
    saved = np.ascontiguousarray(bgr[y1:y2, x1:x2]) if effective_roi.get("enabled") else np.ascontiguousarray(bgr)
    if saved.size == 0:
        raise ValueError("RGB采集 ROI 裁剪结果为空")
    ok, encoded = cv2.imencode(".jpg", saved, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("同步RGB图片编码失败")
    rgb_meta = {
        "encoding": "jpeg",
        "pixel_format_source": "RGB888",
        "source_resolution": {"width": int(frame.width), "height": int(frame.height)},
        "saved_resolution": {"width": int(saved.shape[1]), "height": int(saved.shape[0])},
        "source_stride_bytes": int(frame.stride_bytes),
        "sequence": int(frame.sequence),
        "timestamp_epoch_ms": int(frame.timestamp_epoch_ms),
        "roi": effective_roi,
    }
    return encoded.tobytes(), effective_roi, rgb_meta


def _encode_shared_depth(
    frame: Any,
    capture_roi: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Decode uint16-mm shared depth, apply the same normalized ROI, and encode PNG16."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    expected = int(frame.stride_bytes) * int(frame.height)
    if len(frame.data) < expected:
        raise ValueError("Depth共享帧数据长度不足")
    raw_rows = np.frombuffer(frame.data, dtype=np.uint8, count=expected).reshape(frame.height, frame.stride_bytes)
    compact = np.ascontiguousarray(raw_rows[:, : frame.width * 2])
    depth = compact.view(np.uint16).reshape(frame.height, frame.width)
    effective_roi = resolve_capture_roi_for_image(capture_roi, frame.width, frame.height)
    x1, y1, x2, y2 = effective_roi["pixel_xyxy"]
    saved = np.ascontiguousarray(depth[y1:y2, x1:x2]) if effective_roi.get("enabled") else np.ascontiguousarray(depth)
    if saved.size == 0:
        raise ValueError("Depth采集 ROI 裁剪结果为空")
    ok, encoded = cv2.imencode(".png", saved, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("16位深度 PNG 编码失败")

    saved_intrinsics = {
        "fx": float(frame.fx),
        "fy": float(frame.fy),
        "cx": float(frame.cx) - float(x1),
        "cy": float(frame.cy) - float(y1),
        "width": int(saved.shape[1]),
        "height": int(saved.shape[0]),
    }
    depth_meta = {
        "encoding": "16UC1_PNG",
        "unit": "mm",
        "source_resolution": {"width": int(frame.width), "height": int(frame.height)},
        "saved_resolution": {"width": int(saved.shape[1]), "height": int(saved.shape[0])},
        "source_stride_bytes": int(frame.stride_bytes),
        "sequence": int(frame.sequence),
        "timestamp_epoch_ms": int(frame.timestamp_epoch_ms),
        "aligned_to_color": bool(frame.aligned_to_color),
        "calibration_ready": bool(frame.calibration_ready),
        "flip_horizontal": bool(frame.flip_horizontal),
        "flip_vertical": bool(frame.flip_vertical),
        "intrinsics_source": {
            "fx": float(frame.fx),
            "fy": float(frame.fy),
            "cx": float(frame.cx),
            "cy": float(frame.cy),
            "width": int(frame.width),
            "height": int(frame.height),
        },
        "intrinsics_saved": saved_intrinsics,
        "roi": effective_roi,
        "valid_pixel_count": int(np.count_nonzero(saved)),
        "total_pixel_count": int(saved.size),
    }
    depth_meta["valid_pixel_ratio"] = (
        float(depth_meta["valid_pixel_count"]) / float(depth_meta["total_pixel_count"])
        if depth_meta["total_pixel_count"]
        else 0.0
    )
    return encoded.tobytes(), depth_meta


def _save_synchronized_rgbd(prefix: str, capture_roi: dict[str, Any]) -> dict[str, Any]:
    try:
        bundle = capture_synchronized_rgbd()
    except RgbdCaptureUnavailable as error:
        raise RuntimeUnavailable(str(error)) from error

    rgb_body, effective_rgb_roi, rgb_meta = _encode_shared_rgb(bundle["rgb"], capture_roi)
    depth_body, depth_meta = _encode_shared_depth(bundle["depth"], capture_roi)
    stem = f"{prefix}_{_now_stamp()}"
    image_path = IMAGE_DIR / f"{stem}.jpg"
    depth_path = DEPTH_DIR / f"{stem}.png"
    meta_path = META_DIR / f"{stem}.json"
    captured_at = datetime.fromtimestamp(
        float(bundle["timestamp_epoch_ms"]) / 1000.0,
        tz=timezone.utc,
    ).isoformat()
    metadata = {
        "schema_version": "1.0",
        "message_type": "rgbd_capture_meta",
        "capture_id": stem,
        "captured_at_utc": captured_at,
        "timestamp_epoch_ms": int(bundle["timestamp_epoch_ms"]),
        "synchronized": bool(bundle.get("synchronized")),
        "synchronization_mode": str(bundle.get("synchronization_mode") or ""),
        "camera": {
            "camera_model": bundle.get("camera", {}).get("camera_model"),
            "display_name": bundle.get("camera", {}).get("display_name"),
            "base_url": bundle.get("camera", {}).get("base_url"),
            "selection_path": bundle.get("camera", {}).get("selection_path"),
        },
        "capture_roi": capture_roi,
        "rgb": {"filename": image_path.name, **rgb_meta},
        "depth": {"filename": depth_path.name, **depth_meta},
        "transport": {
            "rgb": "posix_shared_memory",
            "depth": "posix_shared_memory",
            "rgb_source": bundle.get("rgb_shm_path"),
            "depth_source": bundle.get("depth_shm_path"),
        },
    }

    # Write companions first and publish the RGB image last.  The image list is
    # therefore never able to expose a half-written RGB-D record.
    try:
        depth_path.write_bytes(depth_body)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        image_path.write_bytes(rgb_body)
    except Exception:
        image_path.unlink(missing_ok=True)
        depth_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise

    return {
        "schema_version": "1.0",
        "message_type": "dataset_capture_result",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "image": _image_record(image_path),
        "depth": {
            "filename": depth_path.name,
            "path": str(depth_path),
            "size_bytes": depth_path.stat().st_size,
        },
        "meta": {
            "filename": meta_path.name,
            "path": str(meta_path),
            "size_bytes": meta_path.stat().st_size,
        },
        "image_dir": str(IMAGE_DIR),
        "depth_dir": str(DEPTH_DIR),
        "meta_dir": str(META_DIR),
        "content_type": "image/jpeg",
        "depth_saved": True,
        "synchronized": True,
        "synchronization_mode": metadata["synchronization_mode"],
        "images_are_cropped": bool(effective_rgb_roi.get("enabled")),
        "capture_roi": effective_rgb_roi,
    }


def save_runtime_snapshot(
    runtime_client: RuntimeClient,
    prefix: str = "visionops",
    save_depth: bool = False,
) -> dict[str, Any]:
    _ensure_dirs()
    if save_depth:
        with _DATASET_LOCK:
            return _save_synchronized_rgbd(prefix, load_capture_roi_config())

    response = runtime_client.request("GET", f"/api/runtime/snapshot.jpg?t={timestamp_ms()}")
    if response.status_code != 200 or not response.body:
        raise RuntimeUnavailable(f"Runtime snapshot failed: HTTP {response.status_code}")

    with _DATASET_LOCK:
        capture_roi = load_capture_roi_config()
        ext = ".png" if response.content_type == "image/png" else ".jpg"
        body = response.body
        effective_roi = capture_roi
        if capture_roi.get("enabled") is True:
            body, ext, effective_roi = _encode_cropped_snapshot(body, response.content_type, capture_roi)

        filename = f"{prefix}_{_now_stamp()}{ext}"
        path = IMAGE_DIR / filename
        path.write_bytes(body)
        return {
            "schema_version": "1.0",
            "message_type": "dataset_capture_result",
            "status": "ok",
            "timestamp_ms": timestamp_ms(),
            "image": _image_record(path),
            "image_dir": str(IMAGE_DIR),
            "content_type": "image/png" if ext == ".png" else "image/jpeg",
            "depth_saved": False,
            "images_are_cropped": bool(effective_roi.get("enabled")),
            "capture_roi": effective_roi,
        }


def _package_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ms": int(stat.st_mtime * 1000),
        "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def list_packages(limit: int = 20) -> dict[str, Any]:
    _ensure_dirs()
    files = [p for p in PACKAGE_DIR.iterdir() if p.is_file() and p.name.endswith(".tar.gz")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "schema_version": "1.0",
        "message_type": "dataset_package_list",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "package_dir": str(PACKAGE_DIR),
        "packages": [_package_record(p) for p in files[:max(1, min(100, int(limit or 20)))]],
    }


def create_dataset_package(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    # 打包期间暂停同进程内的定时/手动写图，确保图片集合和 capture_manifest 一致。
    with _DATASET_LOCK:
        return _create_dataset_package_locked(metadata)


def _create_dataset_package_locked(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    images = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError("没有可打包的图片，请先拍照采集")

    metadata = metadata if isinstance(metadata, dict) else {}
    device_id = _safe_id(str(metadata.get("device_id") or "rk3576-001"), "rk3576-001")
    customer_id = _safe_id(str(metadata.get("customer_id") or "CUST-001"), "CUST-001")
    contact_info = str(metadata.get("contact_info") or "").strip()
    remark = str(metadata.get("remark") or "").strip()
    created_at, created_stamp = _created_at_now()
    package_name = f"{device_id}_{customer_id}_{created_stamp}.tar.gz"
    package_path = PACKAGE_DIR / package_name

    suffix = 1
    while package_path.exists():
        package_name = f"{device_id}_{customer_id}_{created_stamp}_{suffix:02d}.tar.gz"
        package_path = PACKAGE_DIR / package_name
        suffix += 1

    capture_roi = load_capture_roi_config()
    records: list[dict[str, Any]] = []
    depth_files: list[Path] = []
    meta_files: list[Path] = []
    for image in images:
        depth_path = DEPTH_DIR / f"{image.stem}.png"
        meta_path = META_DIR / f"{image.stem}.json"
        has_depth = depth_path.is_file()
        has_meta = meta_path.is_file()
        if has_depth:
            depth_files.append(depth_path)
        if has_meta:
            meta_files.append(meta_path)
        records.append(
            {
                "capture_id": image.stem,
                "rgb": f"images/{image.name}",
                "depth": f"depth/{depth_path.name}" if has_depth else None,
                "meta": f"meta/{meta_path.name}" if has_meta else None,
                "has_depth": has_depth,
                "has_meta": has_meta,
            }
        )
    capture_manifest = {
        "schema_version": "1.0",
        "message_type": "capture_manifest",
        "created_at": created_at,
        "images_are_cropped": bool(capture_roi.get("enabled")),
        "coordinate_space": "runtime_snapshot",
        "capture_roi": capture_roi,
        "image_count": len(images),
        "rgbd_count": sum(1 for item in records if item["has_depth"] and item["has_meta"]),
        "depth_count": len(depth_files),
        "meta_count": len(meta_files),
        "records": records,
    }
    manifest = {
        "device_id": device_id,
        "customer_id": customer_id,
        "contact_info": contact_info,
        "remark": remark,
        "created_at": created_at,
        "counts": {
            "all": len(images),
            "rgb": len(images),
            "depth": len(depth_files),
            "meta": len(meta_files),
            "rgbd": capture_manifest["rgbd_count"],
        },
        "package_name": package_name,
        "capture": {
            "images_are_cropped": capture_manifest["images_are_cropped"],
            "capture_roi": capture_roi,
            "manifest_file": "capture_manifest.json",
            "rgbd_count": capture_manifest["rgbd_count"],
            "depth_dir": "depth",
            "meta_dir": "meta",
        },
    }
    with tarfile.open(package_path, "w:gz") as tar:
        import io

        for archive_name, document in (
            ("manifest.json", manifest),
            ("capture_manifest.json", capture_manifest),
        ):
            document_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
            info = tarfile.TarInfo(archive_name)
            info.size = len(document_bytes)
            info.mtime = time.time()
            tar.addfile(info, io.BytesIO(document_bytes))
        for path in images:
            tar.add(path, arcname=f"images/{path.name}")
        for path in depth_files:
            tar.add(path, arcname=f"depth/{path.name}")
        for path in meta_files:
            tar.add(path, arcname=f"meta/{path.name}")
    return {
        "schema_version": "1.0",
        "message_type": "dataset_package_create_result",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "package": _package_record(package_path),
        "image_count": len(images),
        "depth_count": len(depth_files),
        "meta_count": len(meta_files),
        "rgbd_count": capture_manifest["rgbd_count"],
        "image_dir": str(IMAGE_DIR),
        "depth_dir": str(DEPTH_DIR),
        "meta_dir": str(META_DIR),
        "package_dir": str(PACKAGE_DIR),
        "manifest": manifest,
        "capture_manifest": capture_manifest,
    }


def _validate_upload_config(config: dict[str, Any]) -> dict[str, Any]:
    upload = config.get("upload") if isinstance(config, dict) else {}
    upload = upload if isinstance(upload, dict) else {}
    server_ip = str(upload.get("server_ip") or "").strip()
    ssh_user = str(upload.get("ssh_user") or "").strip()
    if not server_ip or not ssh_user:
        raise ValueError("请先在视觉盒子设置中配置服务端 IP 和 SSH 用户")
    return {
        "server_ip": server_ip,
        "ssh_user": ssh_user,
        "ssh_password": str(upload.get("ssh_password") or ""),
        "ssh_port": int(upload.get("ssh_port") or 22),
        "remote_dir": str(upload.get("remote_dir") or "/opt/visionops_uploads").strip() or "/opt/visionops_uploads",
        "timeout_s": int(upload.get("timeout_s") or 60),
    }


def _upload_with_paramiko(package_path: Path, upload: dict[str, Any]) -> dict[str, Any] | None:
    if importlib.util.find_spec("paramiko") is None:
        return None
    import paramiko  # type: ignore

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        upload["server_ip"],
        port=upload["ssh_port"],
        username=upload["ssh_user"],
        password=upload["ssh_password"] or None,
        timeout=upload["timeout_s"],
        banner_timeout=upload["timeout_s"],
        auth_timeout=upload["timeout_s"],
    )
    try:
        command = "mkdir -p " + _shell_quote(upload["remote_dir"])
        _stdin, stdout, stderr = client.exec_command(command, timeout=upload["timeout_s"])
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(stderr.read().decode("utf-8", errors="ignore") or f"mkdir failed: {exit_code}")
        sftp = client.open_sftp()
        try:
            remote_path = upload["remote_dir"].rstrip("/") + "/" + package_path.name
            sftp.put(str(package_path), remote_path)
        finally:
            sftp.close()
        return {"method": "paramiko", "remote_path": remote_path}
    finally:
        client.close()


def _shell_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def _run_upload_command(args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)


def _upload_with_system_tools(package_path: Path, upload: dict[str, Any]) -> dict[str, Any]:
    ssh = shutil.which("ssh")
    scp = shutil.which("scp")
    if not ssh or not scp:
        raise RuntimeError("系统缺少 ssh/scp，且未安装 paramiko")
    base_ssh = [ssh, "-p", str(upload["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    base_scp = [scp, "-P", str(upload["ssh_port"]), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if upload["ssh_password"]:
        sshpass = shutil.which("sshpass")
        if not sshpass:
            raise RuntimeError("使用密码上传需要安装 sshpass，或在 Collector venv 中安装 paramiko")
        base_ssh = [sshpass, "-p", upload["ssh_password"], *base_ssh]
        base_scp = [sshpass, "-p", upload["ssh_password"], *base_scp]
    remote_user_host = f"{upload['ssh_user']}@{upload['server_ip']}"
    mkdir_cmd = [*base_ssh, remote_user_host, "mkdir", "-p", upload["remote_dir"]]
    timeout = max(5, int(upload["timeout_s"]))
    mkdir = _run_upload_command(mkdir_cmd, timeout)
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr.strip() or mkdir.stdout.strip() or f"mkdir failed: {mkdir.returncode}")
    remote_path = upload["remote_dir"].rstrip("/") + "/" + package_path.name
    scp_cmd = [*base_scp, str(package_path), f"{remote_user_host}:{remote_path}"]
    scp_result = _run_upload_command(scp_cmd, timeout)
    if scp_result.returncode != 0:
        raise RuntimeError(scp_result.stderr.strip() or scp_result.stdout.strip() or f"scp failed: {scp_result.returncode}")
    return {"method": "scp", "remote_path": remote_path}


def create_and_upload_dataset(config: CollectorConfig, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    package_result = create_dataset_package(metadata)
    package_path = Path(package_result["package"]["path"])
    started = time.perf_counter()
    upload: dict[str, Any] = {}
    try:
        settings = load_vision_box_settings(config)
        upload = _validate_upload_config(settings)
        upload_result = _upload_with_paramiko(package_path, upload)
        if upload_result is None:
            upload_result = _upload_with_system_tools(package_path, upload)
    except Exception as error:  # noqa: BLE001 - return package path for retry/debug
        return {
            "schema_version": "1.0",
            "message_type": "dataset_upload_result",
            "status": "error",
            "upload_ok": False,
            "timestamp_ms": timestamp_ms(),
            "package": package_result["package"],
            "image_count": package_result["image_count"],
            "manifest": package_result.get("manifest"),
            "capture_manifest": package_result.get("capture_manifest"),
            "upload": {
                "server_ip": upload.get("server_ip"),
                "ssh_user": upload.get("ssh_user"),
                "ssh_port": upload.get("ssh_port"),
                "remote_dir": upload.get("remote_dir"),
                "error": str(error),
            },
            "message": "上传失败，压缩包已保留在本地，可稍后重试或手动拷贝。",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "schema_version": "1.0",
        "message_type": "dataset_upload_result",
        "status": "ok",
        "upload_ok": True,
        "timestamp_ms": timestamp_ms(),
        "package": package_result["package"],
        "image_count": package_result["image_count"],
        "manifest": package_result.get("manifest"),
        "capture_manifest": package_result.get("capture_manifest"),
        "upload": {
            "server_ip": upload.get("server_ip"),
            "ssh_user": upload.get("ssh_user"),
            "ssh_port": upload.get("ssh_port"),
            "remote_dir": upload.get("remote_dir"),
            **upload_result,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
