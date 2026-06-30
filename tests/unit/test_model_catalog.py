from __future__ import annotations

import json
from pathlib import Path

from apps.collector_web.backend.model_catalog import find_scanned_model, scan_model_catalog


def _write_model_package(root: Path, name: str, *, include_rknn: bool = True) -> Path:
    package = root / name
    package.mkdir(parents=True)
    manifest = {
        "package_id": f"{name}-id",
        "model_name": name,
        "model_version": "0.1.0",
        "task_type": "obb",
        "target_platform": "rk3576",
        "files": {
            "rknn": "model.rknn",
            "yaml": "model.yaml",
            "labels": "labels.txt",
        },
        "input": {"size": [640, 640]},
        "postprocess": {"score_threshold": 0.5, "nms_threshold": 0.45},
    }
    (package / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (package / "model.yaml").write_text(
        "model_name: %s\nmodel_version: 0.1.0\ntask_type: obb\ninput_size: [640, 640]\nclass_names: [tube, defect]\n"
        % name,
        encoding="utf-8",
    )
    (package / "labels.txt").write_text("tube\ndefect\n", encoding="utf-8")
    if include_rknn:
        (package / "model.rknn").write_bytes(b"mock-rknn")
    return package


def test_scan_model_catalog_recognizes_standard_package(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "carton_tube_check")
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    model = models[0]
    assert model["valid"] is True
    assert model["package_dir"] == "carton_tube_check"
    assert model["task_type"] == "obb"
    assert model["input_size"] == [640, 640]
    assert model["labels_count"] == 2


def test_scan_model_catalog_marks_missing_rknn_invalid(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "broken_model", include_rknn=False)
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    assert models[0]["valid"] is False
    assert "rknn 文件不存在" in models[0]["error"]


def test_scan_model_catalog_ignores_directory_without_manifest(tmp_path: Path) -> None:
    package = _write_model_package(tmp_path, "carton_partition_check")
    (package / "manifest.json").unlink()
    models = scan_model_catalog(tmp_path)
    assert models == []


def test_scan_model_catalog_does_not_treat_extra_model2_as_second_package(tmp_path: Path) -> None:
    package = _write_model_package(tmp_path, "carton_tube_check")
    (package / "model2.rknn").write_bytes(b"another")
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    assert models[0]["rknn_file"] == "model.rknn"


def test_find_scanned_model_only_matches_known_entries(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "carton_tube_check")
    models = scan_model_catalog(tmp_path)
    assert find_scanned_model(models, package_dir="carton_tube_check") is not None
    assert find_scanned_model(models, model_id="carton_tube_check-id") is not None
    assert find_scanned_model(models, package_dir="../../etc") is None


def test_scan_model_catalog_rejects_manifest_path_escape(tmp_path: Path) -> None:
    package = tmp_path / "escape_model"
    package.mkdir(parents=True)
    (tmp_path / "outside.rknn").write_bytes(b"outside")
    (package / "model.yaml").write_text("model_name: escape\n", encoding="utf-8")
    (package / "labels.txt").write_text("object\n", encoding="utf-8")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "package_id": "escape-id",
                "model_name": "escape",
                "model_version": "0.0.1",
                "task_type": "detection",
                "files": {"rknn": "../outside.rknn", "yaml": "model.yaml", "labels": "labels.txt"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    assert models[0]["valid"] is False
