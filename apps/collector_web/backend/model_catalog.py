"""Collector Web 模型目录扫描与受控选择。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelCatalogItem:
    """扫描到的一级模型包摘要。"""

    model_id: str
    package_dir: str
    package_path: str
    model_name: str
    model_version: str
    task_type: str
    target_platform: str
    input_size: list[int]
    rknn_file: str
    yaml_file: str
    labels_file: str
    rknn_size_bytes: int
    labels_count: int
    valid: bool
    active: bool
    mtime_ms: int | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        document = {
            "model_id": self.model_id,
            "package_dir": self.package_dir,
            "package_path": self.package_path,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task_type": self.task_type,
            "target_platform": self.target_platform,
            "input_size": self.input_size,
            "rknn_file": self.rknn_file,
            "yaml_file": self.yaml_file,
            "labels_file": self.labels_file,
            "rknn_size_bytes": self.rknn_size_bytes,
            "labels_count": self.labels_count,
            "valid": self.valid,
            "active": self.active,
            "mtime_ms": self.mtime_ms,
        }
        if self.error:
            document["error"] = self.error
        return document


def default_models_root(project_root: Path) -> Path:
    """优先使用仓库根目录 models，其次使用 /opt/visionops_v3/models。"""

    repo_models = project_root / "models"
    if repo_models.exists():
        return repo_models
    return Path("/opt/visionops_v3/models")


def scan_model_catalog(models_root: Path, current_model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """扫描 models_root 下的一级模型包目录。"""

    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return []
    current_rknn = _normalized_path(current_model.get("rknn_path")) if isinstance(current_model, dict) else None
    current_model_id = str(current_model.get("model_id", "")) if isinstance(current_model, dict) else ""
    items: list[dict[str, Any]] = []
    for package_dir in sorted((entry for entry in root.iterdir() if entry.is_dir()), key=lambda value: value.name):
        item = _scan_package_dir(package_dir, current_model_id=current_model_id, current_rknn=current_rknn)
        if item is not None:
            items.append(item.as_dict())
    return items


def find_scanned_model(
    models: list[dict[str, Any]],
    *,
    model_id: str | None = None,
    package_dir: str | None = None,
) -> dict[str, Any] | None:
    """只允许通过已扫描到的模型包进行选择。"""

    selected_model_id = (model_id or "").strip()
    selected_package_dir = (package_dir or "").strip()
    for model in models:
        if selected_model_id and model.get("model_id") == selected_model_id:
            return model
        if selected_package_dir and model.get("package_dir") == selected_package_dir:
            return model
    return None


def _scan_package_dir(
    package_dir: Path,
    *,
    current_model_id: str,
    current_rknn: str | None,
) -> ModelCatalogItem | None:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _invalid_model(package_dir, f"manifest.json 解析失败: {error}")
    if not isinstance(manifest, dict):
        return _invalid_model(package_dir, "manifest.json 顶层必须是对象")

    files = manifest.get("files")
    if not isinstance(files, dict):
        return _invalid_model(package_dir, "manifest.json 缺少 files 对象")

    rknn_file = _safe_manifest_file(package_dir, files.get("rknn"))
    yaml_file = _safe_manifest_file(package_dir, files.get("yaml"))
    labels_file = _safe_manifest_file(package_dir, files.get("labels"))
    missing = []
    for name, value in (("rknn", rknn_file), ("yaml", yaml_file), ("labels", labels_file)):
        if value is None:
            missing.append(name)
    if missing:
        return _invalid_model(package_dir, f"manifest.json 缺少或越界文件字段: {', '.join(missing)}")

    file_errors = []
    for label, path in (("rknn", rknn_file), ("yaml", yaml_file), ("labels", labels_file)):
        if not path.is_file():
            file_errors.append(f"{label} 文件不存在: {path.name}")
    if file_errors:
        return _invalid_model(package_dir, "; ".join(file_errors), rknn_file.name, yaml_file.name, labels_file.name)

    yaml_meta = _load_yaml_meta(yaml_file)
    manifest_input_size = _manifest_input_size(manifest)
    input_size = manifest_input_size or yaml_meta["input_size"] or [640, 640]
    labels_count = _count_labels(labels_file) or yaml_meta["labels_count"]
    package_path = str(package_dir.resolve())
    active = False
    rknn_path = _normalized_path(str(rknn_file.resolve()))
    model_id = str(manifest.get("package_id") or yaml_meta["model_name"] or package_dir.name)
    if current_model_id and model_id == current_model_id:
        active = True
    elif current_rknn and current_rknn == rknn_path:
        active = True

    try:
        rknn_size_bytes = rknn_file.stat().st_size
        mtime_ms = int(rknn_file.stat().st_mtime * 1000)
    except OSError:
        rknn_size_bytes = 0
        mtime_ms = None

    return ModelCatalogItem(
        model_id=model_id,
        package_dir=package_dir.name,
        package_path=package_path,
        model_name=str(manifest.get("model_name") or yaml_meta["model_name"] or package_dir.name),
        model_version=str(manifest.get("model_version") or yaml_meta["model_version"] or "unknown"),
        task_type=str(manifest.get("task_type") or yaml_meta["task_type"] or "detection"),
        target_platform=str(manifest.get("target_platform") or "unknown"),
        input_size=input_size,
        rknn_file=rknn_file.name,
        yaml_file=yaml_file.name,
        labels_file=labels_file.name,
        rknn_size_bytes=rknn_size_bytes,
        labels_count=labels_count,
        valid=True,
        active=active,
        mtime_ms=mtime_ms,
    )


def _invalid_model(
    package_dir: Path,
    error: str,
    rknn_file: str = "",
    yaml_file: str = "",
    labels_file: str = "",
) -> ModelCatalogItem:
    return ModelCatalogItem(
        model_id=package_dir.name,
        package_dir=package_dir.name,
        package_path=str(package_dir.resolve()),
        model_name=package_dir.name,
        model_version="unknown",
        task_type="unknown",
        target_platform="unknown",
        input_size=[640, 640],
        rknn_file=rknn_file,
        yaml_file=yaml_file,
        labels_file=labels_file,
        rknn_size_bytes=0,
        labels_count=0,
        valid=False,
        active=False,
        mtime_ms=None,
        error=error,
    )


def _safe_manifest_file(package_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (package_dir / value).resolve()
    package_root = package_dir.resolve()
    try:
        if candidate.relative_to(package_root):
            return candidate
    except ValueError:
        return None
    return candidate


def _manifest_input_size(manifest: dict[str, Any]) -> list[int] | None:
    input_config = manifest.get("input")
    if not isinstance(input_config, dict):
        return None
    raw = input_config.get("size", input_config.get("input_size"))
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            width = int(raw[0])
            height = int(raw[1])
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return [width, height]
    width = input_config.get("width")
    height = input_config.get("height")
    try:
        if int(width) > 0 and int(height) > 0:
            return [int(width), int(height)]
    except (TypeError, ValueError):
        return None
    return None


def _load_yaml_meta(path: Path) -> dict[str, Any]:
    defaults = {
        "model_name": "",
        "model_version": "",
        "task_type": "",
        "input_size": None,
        "labels_count": 0,
    }
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return defaults
    if not isinstance(document, dict):
        return defaults
    result = dict(defaults)
    result["model_name"] = str(document.get("model_name") or "")
    result["model_version"] = str(document.get("model_version") or "")
    result["task_type"] = str(document.get("task_type") or "")
    input_size = document.get("input_size")
    if isinstance(input_size, list) and len(input_size) >= 2:
        try:
            result["input_size"] = [int(input_size[0]), int(input_size[1])]
        except (TypeError, ValueError):
            result["input_size"] = None
    class_names = document.get("class_names")
    if isinstance(class_names, list):
        result["labels_count"] = len([item for item in class_names if str(item).strip()])
    return result


def _count_labels(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0


def _normalized_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve())
