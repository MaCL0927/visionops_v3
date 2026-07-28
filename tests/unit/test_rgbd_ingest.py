"""Server ingest keeps RGB-D companions without treating depth PNG as annotation images."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from apps.server_api.backend.services.ingest_service import BatchService


def _package(path: Path) -> Path:
    root = path.parent / "rgbd_root"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "meta").mkdir()
    (root / "images" / "frame.jpg").write_bytes(b"rgb")
    (root / "depth" / "frame.png").write_bytes(b"depth16")
    (root / "meta" / "frame.json").write_text("{}", encoding="utf-8")
    capture = {
        "schema_version": "1.0",
        "message_type": "capture_manifest",
        "images_are_cropped": False,
        "image_count": 1,
        "depth_count": 1,
        "meta_count": 1,
        "rgbd_count": 1,
    }
    manifest = {
        "device_id": "rk3576-001",
        "customer_id": "rgbd",
        "counts": {"all": 1, "rgb": 1, "depth": 1, "meta": 1, "rgbd": 1},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "capture_manifest.json").write_text(json.dumps(capture), encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        for rel in (
            "manifest.json",
            "capture_manifest.json",
            "images/frame.jpg",
            "depth/frame.png",
            "meta/frame.json",
        ):
            archive.add(root / rel, arcname=rel)
    return path


def test_ingest_preserves_depth_and_meta_but_counts_one_rgb_image(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    package = _package(incoming / "rk3576-001_rgbd_20260728_120000.tar.gz")
    service = BatchService(tmp_path / "batches", ("segmentation",), incoming_root=incoming)
    batch = service.process_incoming_packages([package.name])

    raw = Path(batch["raw_path"])
    assert batch["image_count"] == 1
    assert (raw / "images" / "frame.jpg").is_file()
    assert (raw / "depth" / "frame.png").is_file()
    assert (raw / "meta" / "frame.json").is_file()
