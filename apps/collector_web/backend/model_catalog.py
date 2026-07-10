"""Collector Web 模型目录扫描与受控选择。

模型包规则固定为：

    models/<model_dir>/model.rknn
    models/<model_dir>/model.yaml

不再读取 manifest.json / labels.txt，也不扫描平铺 rknn/yaml。
model.yaml 是模型列表和 Runtime 切换的唯一元信息来源。
"""

from __future__ import annotations

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
            # 标准模型包不再要求 labels.txt；保留字段是为了兼容已有前端与测试。
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
    """扫描 models_root 下的一级标准模型包目录。"""

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
    rknn_file = package_dir / "model.rknn"
    yaml_file = package_dir / "model.yaml"

    # 非标准目录直接忽略，避免把 Log、临时目录等展示成无效模型。
    if not rknn_file.exists() and not yaml_file.exists():
        return None

    yaml_meta = _load_yaml_meta(yaml_file) if yaml_file.is_file() else _default_yaml_meta()
    file_errors = []
    if not rknn_file.is_file():
        file_errors.append("缺少 model.rknn")
    if not yaml_file.is_file():
        file_errors.append("缺少 model.yaml")
    if yaml_meta.get("error"):
        file_errors.append(str(yaml_meta["error"]))

    package_path = str(package_dir.resolve())
    rknn_path = _normalized_path(str(rknn_file.resolve())) if rknn_file.exists() else None
    model_id = str(yaml_meta.get("model_id") or package_dir.name)
    active = False
    if current_model_id and model_id == current_model_id:
        active = True
    elif current_rknn and rknn_path and current_rknn == rknn_path:
        active = True

    try:
        rknn_size_bytes = rknn_file.stat().st_size if rknn_file.is_file() else 0
        mtime_ms = int(rknn_file.stat().st_mtime * 1000) if rknn_file.is_file() else None
    except OSError:
        rknn_size_bytes = 0
        mtime_ms = None

    return ModelCatalogItem(
        model_id=model_id,
        package_dir=package_dir.name,
        package_path=package_path,
        model_name=str(yaml_meta.get("model_name") or package_dir.name),
        model_version=str(yaml_meta.get("model_version") or "unknown"),
        task_type=str(yaml_meta.get("task_type") or "unknown"),
        target_platform=str(yaml_meta.get("target_platform") or "unknown"),
        input_size=yaml_meta.get("input_size") or [640, 640],
        rknn_file="model.rknn",
        yaml_file="model.yaml",
        labels_file="",
        rknn_size_bytes=rknn_size_bytes,
        labels_count=int(yaml_meta.get("labels_count") or 0),
        valid=not file_errors,
        active=active,
        mtime_ms=mtime_ms,
        error="; ".join(file_errors) if file_errors else None,
    )


def _default_yaml_meta() -> dict[str, Any]:
    return {
        "model_id": "",
        "model_name": "",
        "model_version": "",
        "task_type": "",
        "target_platform": "",
        "input_size": None,
        "labels_count": 0,
        "error": None,
    }


def _load_yaml_meta(path: Path) -> dict[str, Any]:
    result = _default_yaml_meta()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        result["error"] = f"model.yaml 解析失败: {error}"
        return result
    if not isinstance(document, dict):
        result["error"] = "model.yaml 顶层必须是对象"
        return result

    result["model_id"] = str(document.get("model_id") or document.get("package_id") or "")
    result["model_name"] = str(document.get("model_name") or document.get("display_name") or "")
    result["model_version"] = str(document.get("model_version") or document.get("version") or "")
    result["task_type"] = str(document.get("task_type") or document.get("task") or "")
    result["target_platform"] = str(document.get("target_platform") or document.get("platform") or "")

    dataset_section = document.get("dataset")
    if isinstance(dataset_section, dict) and not result["target_platform"]:
        result["target_platform"] = str(dataset_section.get("device_id") or "")

    # 兼容部署 YAML 中常见的嵌套 model.name/display_name，仅作为展示名补充。
    model_section = document.get("model")
    if isinstance(model_section, dict):
        if not result["model_name"]:
            result["model_name"] = str(model_section.get("display_name") or model_section.get("name") or "")
        if not result["model_id"]:
            result["model_id"] = str(model_section.get("id") or "")

    input_size = (
        document.get("input_size")
        or document.get("imgsz")
        or document.get("image_size")
        or document.get("model_input_size")
    )
    parsed_input = _parse_input_size(input_size)
    if parsed_input:
        result["input_size"] = parsed_input

    class_names = document.get("class_names", document.get("names"))
    if isinstance(class_names, dict):
        labels = [str(class_names[key]).strip() for key in sorted(class_names) if str(class_names[key]).strip()]
    elif isinstance(class_names, list):
        labels = [str(item).strip() for item in class_names if str(item).strip()]
    else:
        labels = []
    result["labels_count"] = len(labels)
    return result


def _parse_input_size(raw: Any) -> list[int] | None:
    try:
        if isinstance(raw, int):
            return [raw, raw] if raw > 0 else None
        if isinstance(raw, str):
            value = raw.strip().strip("[]")
            parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
            if len(parts) == 1:
                size = int(parts[0])
                return [size, size] if size > 0 else None
            if len(parts) >= 2:
                width, height = int(parts[0]), int(parts[1])
                return [width, height] if width > 0 and height > 0 else None
        if isinstance(raw, list):
            if len(raw) == 1:
                size = int(raw[0])
                return [size, size] if size > 0 else None
            if len(raw) >= 2:
                width, height = int(raw[0]), int(raw[1])
                return [width, height] if width > 0 and height > 0 else None
    except (TypeError, ValueError):
        return None
    return None


def _normalized_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve())
