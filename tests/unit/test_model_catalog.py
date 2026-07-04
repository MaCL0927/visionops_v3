from __future__ import annotations

from pathlib import Path

from apps.collector_web.backend.model_catalog import find_scanned_model, scan_model_catalog


def _write_model_package(root: Path, name: str, *, include_rknn: bool = True, include_yaml: bool = True) -> Path:
    package = root / name
    package.mkdir(parents=True)
    if include_yaml:
        (package / "model.yaml").write_text(
            "model_id: %s-id\n"
            "model_name: %s\n"
            "model_version: 0.1.0\n"
            "task: obb\n"
            "target_platform: rk3576\n"
            "input_size: [640, 640]\n"
            "class_names:\n"
            "- tube\n"
            "- defect\n"
            "conf_threshold: 0.5\n"
            "nms_threshold: 0.45\n" % (name, name),
            encoding="utf-8",
        )
    if include_rknn:
        (package / "model.rknn").write_bytes(b"mock-rknn")
    return package


def test_scan_model_catalog_recognizes_m15_standard_package(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "carton_tube_check")
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    model = models[0]
    assert model["valid"] is True
    assert model["package_dir"] == "carton_tube_check"
    assert model["model_id"] == "carton_tube_check-id"
    assert model["task_type"] == "obb"
    assert model["target_platform"] == "rk3576"
    assert model["input_size"] == [640, 640]
    assert model["labels_count"] == 2
    assert model["rknn_file"] == "model.rknn"
    assert model["yaml_file"] == "model.yaml"
    assert model["labels_file"] == ""


def test_scan_model_catalog_marks_missing_rknn_invalid(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "broken_model", include_rknn=False)
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    assert models[0]["valid"] is False
    assert "缺少 model.rknn" in models[0]["error"]


def test_scan_model_catalog_marks_missing_yaml_invalid(tmp_path: Path) -> None:
    _write_model_package(tmp_path, "broken_model", include_yaml=False)
    models = scan_model_catalog(tmp_path)
    assert len(models) == 1
    assert models[0]["valid"] is False
    assert "缺少 model.yaml" in models[0]["error"]


def test_scan_model_catalog_ignores_directory_without_model_files(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
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
