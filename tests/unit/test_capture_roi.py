"""M32.1 采集 ROI 链路测试。"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apps.collector_web.backend import capture_roi as roi_module
from apps.collector_web.backend import dataset_manager as dataset_module
from apps.collector_web.backend.runtime_client import RuntimeResponse


class _FakeRuntimeClient:
    def __init__(self, body: bytes, content_type: str = "image/jpeg") -> None:
        self.body = body
        self.content_type = content_type
        self.calls = 0

    def request(self, method: str, path: str) -> RuntimeResponse:
        self.calls += 1
        assert method == "GET"
        assert path.startswith("/api/runtime/snapshot.jpg")
        return RuntimeResponse(
            status_code=200,
            content_type=self.content_type,
            body=self.body,
            headers={},
        )


def _configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    meta_dir = tmp_path / "meta"
    package_dir = tmp_path / "packages"
    roi_path = tmp_path / "capture_roi.json"
    monkeypatch.setattr(dataset_module, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(dataset_module, "DEPTH_DIR", depth_dir)
    monkeypatch.setattr(dataset_module, "META_DIR", meta_dir)
    monkeypatch.setattr(dataset_module, "PACKAGE_DIR", package_dir)
    monkeypatch.setattr(dataset_module, "CAPTURE_ROI_CONFIG_PATH", roi_path)
    monkeypatch.setattr(roi_module, "CAPTURE_ROI_CONFIG_PATH", roi_path)
    return image_dir, package_dir, roi_path


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    # 采用渐变图，便于确认裁剪后不是空白占位数据。
    x = np.linspace(0, 255, width, dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    image[:, :, 2] = 120
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def test_capture_roi_is_canonicalized_to_source_pixels() -> None:
    result = roi_module.normalize_capture_roi(
        {
            "enabled": True,
            "source_resolution": {"width": 1280, "height": 720},
            "normalized_xyxy": [0.25, 0.2, 0.75, 0.8],
        }
    )
    assert result["enabled"] is True
    assert result["pixel_xyxy"] == [320, 144, 960, 576]
    assert result["crop_resolution"] == {"width": 640, "height": 432}


def test_manual_snapshot_is_saved_as_roi_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_dir, _, _ = _configure_paths(monkeypatch, tmp_path)
    dataset_module.update_capture_roi(
        {
            "enabled": True,
            "source_resolution": {"width": 200, "height": 100},
            "normalized_xyxy": [0.25, 0.2, 0.75, 0.8],
        }
    )
    result = dataset_module.save_runtime_snapshot(_FakeRuntimeClient(_jpeg()))
    saved = image_dir / result["image"]["filename"]
    decoded = cv2.imread(str(saved), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (60, 100)
    assert result["images_are_cropped"] is True
    assert result["capture_roi"]["pixel_xyxy"] == [50, 20, 150, 80]


def test_roi_change_is_blocked_until_existing_batch_is_cleared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_dir, _, _ = _configure_paths(monkeypatch, tmp_path)
    image_dir.mkdir(parents=True)
    (image_dir / "existing.jpg").write_bytes(_jpeg())

    with pytest.raises(dataset_module.CaptureRoiConflict):
        dataset_module.update_capture_roi(
            {
                "enabled": True,
                "source_resolution": {"width": 200, "height": 100},
                "normalized_xyxy": [0.1, 0.1, 0.9, 0.9],
            }
        )

    result = dataset_module.update_capture_roi(
        {
            "enabled": True,
            "source_resolution": {"width": 200, "height": 100},
            "normalized_xyxy": [0.1, 0.1, 0.9, 0.9],
            "clear_existing_images": True,
        }
    )
    assert result["deleted_image_count"] == 1
    assert list(image_dir.iterdir()) == []


def test_dataset_package_contains_capture_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_dir, _, _ = _configure_paths(monkeypatch, tmp_path)
    dataset_module.update_capture_roi(
        {
            "enabled": True,
            "source_resolution": {"width": 200, "height": 100},
            "normalized_xyxy": [0.25, 0.2, 0.75, 0.8],
        }
    )
    dataset_module.save_runtime_snapshot(_FakeRuntimeClient(_jpeg()))
    result = dataset_module.create_dataset_package({"device_id": "rk3576-test", "customer_id": "roi"})

    with tarfile.open(result["package"]["path"], "r:gz") as archive:
        names = set(archive.getnames())
        assert "manifest.json" in names
        assert "capture_manifest.json" in names
        capture_file = archive.extractfile("capture_manifest.json")
        assert capture_file is not None
        capture_manifest = json.load(io.TextIOWrapper(capture_file, encoding="utf-8"))
        assert capture_manifest["images_are_cropped"] is True
        assert capture_manifest["capture_roi"]["pixel_xyxy"] == [50, 20, 150, 80]
        assert len([name for name in names if name.startswith("images/")]) == 1


def _fake_rgbd_bundle(width: int = 200, height: int = 100) -> dict:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180
    rgb[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[:, :, 2] = 40
    depth = np.full((height, width), 1234, dtype=np.uint16)
    timestamp = 1785200000123
    return {
        "camera": {
            "camera_model": "orbbec336l",
            "display_name": "Orbbec Gemini 336L",
            "base_url": "http://127.0.0.1:18182",
            "selection_path": "/tmp/active_camera.json",
        },
        "synchronized": True,
        "synchronization_mode": "posix_shared_memory_timestamp_match",
        "timestamp_epoch_ms": timestamp,
        "rgb_shm_path": "/dev/shm/test_rgb",
        "depth_shm_path": "/dev/shm/test_depth",
        "rgb": SimpleNamespace(
            data=rgb.tobytes(),
            width=width,
            height=height,
            stride_bytes=width * 3,
            sequence=11,
            timestamp_epoch_ms=timestamp,
        ),
        "depth": SimpleNamespace(
            data=depth.tobytes(),
            width=width,
            height=height,
            stride_bytes=width * 2,
            sequence=11,
            timestamp_epoch_ms=timestamp,
            aligned_to_color=True,
            calibration_ready=True,
            flip_horizontal=False,
            flip_vertical=False,
            fx=120.0,
            fy=121.0,
            cx=100.0,
            cy=50.0,
        ),
    }


def test_synchronized_rgbd_capture_saves_rgb_depth_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_dir, _, _ = _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dataset_module, "capture_synchronized_rgbd", _fake_rgbd_bundle)
    dataset_module.update_capture_roi(
        {
            "enabled": True,
            "source_resolution": {"width": 200, "height": 100},
            "normalized_xyxy": [0.25, 0.2, 0.75, 0.8],
        }
    )

    result = dataset_module.save_runtime_snapshot(
        _FakeRuntimeClient(_jpeg()),
        save_depth=True,
    )
    image_path = image_dir / result["image"]["filename"]
    depth_path = dataset_module.DEPTH_DIR / result["depth"]["filename"]
    meta_path = dataset_module.META_DIR / result["meta"]["filename"]

    rgb_saved = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth_saved = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert rgb_saved is not None and rgb_saved.shape[:2] == (60, 100)
    assert depth_saved is not None and depth_saved.dtype == np.uint16
    assert depth_saved.shape == (60, 100)
    assert int(depth_saved[0, 0]) == 1234
    assert result["depth_saved"] is True
    assert result["synchronized"] is True
    assert metadata["rgb"]["sequence"] == 11
    assert metadata["depth"]["sequence"] == 11
    assert metadata["depth"]["intrinsics_saved"]["cx"] == pytest.approx(50.0)
    assert metadata["depth"]["intrinsics_saved"]["cy"] == pytest.approx(30.0)


def test_rgbd_files_are_packaged_and_deleted_with_rgb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dataset_module, "capture_synchronized_rgbd", _fake_rgbd_bundle)
    captured = dataset_module.save_runtime_snapshot(_FakeRuntimeClient(_jpeg()), save_depth=True)
    filename = captured["image"]["filename"]
    stem = Path(filename).stem

    package = dataset_module.create_dataset_package(
        {"device_id": "rk3576-test", "customer_id": "rgbd"}
    )
    assert package["rgbd_count"] == 1
    with tarfile.open(package["package"]["path"], "r:gz") as archive:
        names = set(archive.getnames())
        assert f"images/{filename}" in names
        assert f"depth/{stem}.png" in names
        assert f"meta/{stem}.json" in names
        capture_file = archive.extractfile("capture_manifest.json")
        assert capture_file is not None
        capture_manifest = json.load(io.TextIOWrapper(capture_file, encoding="utf-8"))
        assert capture_manifest["rgbd_count"] == 1
        assert capture_manifest["records"][0]["has_depth"] is True

    deleted = dataset_module.delete_image(filename)
    assert len(deleted["deleted_companions"]) == 2
    assert not (dataset_module.DEPTH_DIR / f"{stem}.png").exists()
    assert not (dataset_module.META_DIR / f"{stem}.json").exists()
