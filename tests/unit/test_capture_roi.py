"""M32.1 采集 ROI 链路测试。"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

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
    package_dir = tmp_path / "packages"
    roi_path = tmp_path / "capture_roi.json"
    monkeypatch.setattr(dataset_module, "IMAGE_DIR", image_dir)
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
