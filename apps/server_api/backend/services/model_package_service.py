"""v3 标准模型包生成、扫描和发布。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def normalize_model_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg"}:
        return "segmentation"
    if task in {"classification", "cls", "classify"}:
        return "classification"
    return "detection"


def make_model_yaml(
    *,
    model_id: str,
    model_name: str,
    version: str,
    task_type: str,
    classes: list[dict[str, Any]] | list[str] | None = None,
    input_size: list[int] | tuple[int, int] = (640, 640),
    target_platform: str = "rk3576",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 100,
    preprocess: str = "letterbox",
    color: str = "rgb",
) -> dict[str, Any]:
    task_type = normalize_model_task(task_type)
    if task_type == "classification" and preprocess == "letterbox":
        preprocess = "resize"
    normalized_classes: list[dict[str, Any]] = []
    if classes:
        for index, item in enumerate(classes):
            if isinstance(item, dict):
                normalized_classes.append({"id": int(item.get("id", index)), "name": str(item.get("name", f"class_{index}"))})
            else:
                normalized_classes.append({"id": index, "name": str(item)})
    if not normalized_classes:
        normalized_classes = [{"id": 0, "name": "object"}]
    width, height = int(input_size[0]), int(input_size[1])
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": model_name,
        "model_version": version,
        "task_type": task_type,
        "target_platform": target_platform,
        "input_size": [width, height],
        "model": {
            "name": model_name,
            "version": version,
            "task": task_type,
            "format": "rknn",
            "target_platform": target_platform,
            "input_size": [width, height],
        },
        "classes": normalized_classes,
        "class_names": [item["name"] for item in normalized_classes],
        "postprocess": {
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "max_det": int(max_det),
        },
        "runtime": {
            "preprocess": preprocess,
            "color": color,
        },
    }


def _yaml_scalar(value: Any) -> str:
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "[", "]", "{", "}", ",", "&", "*", "!", "|", ">", "'", '"']) or text.lower() in {"true", "false", "null", "none"}:
        return "'" + text.replace("'", "''") + "'"
    return text


def write_model_yaml(path: Path, document: dict[str, Any]) -> None:
    size = document.get("input_size") or [640, 640]
    width, height = int(size[0]), int(size[1])
    model = document.get("model") if isinstance(document.get("model"), dict) else {}
    post = document.get("postprocess") if isinstance(document.get("postprocess"), dict) else {}
    runtime = document.get("runtime") if isinstance(document.get("runtime"), dict) else {}
    classes = document.get("classes") if isinstance(document.get("classes"), list) else []
    class_names = document.get("class_names") if isinstance(document.get("class_names"), list) else []

    lines: list[str] = [
        "schema_version: '1.0'",
        f"model_id: {_yaml_scalar(document.get('model_id', ''))}",
        f"model_name: {_yaml_scalar(document.get('model_name', ''))}",
        f"model_version: {_yaml_scalar(document.get('model_version', ''))}",
        f"task_type: {_yaml_scalar(document.get('task_type', 'detection'))}",
        f"target_platform: {_yaml_scalar(document.get('target_platform', 'rk3576'))}",
        f"input_size: [{width}, {height}]",
        "model:",
        f"  name: {_yaml_scalar(model.get('name', document.get('model_name', '')))}",
        f"  version: {_yaml_scalar(model.get('version', document.get('model_version', '')))}",
        f"  task: {_yaml_scalar(model.get('task', document.get('task_type', 'detection')))}",
        f"  format: {_yaml_scalar(model.get('format', 'rknn'))}",
        f"  target_platform: {_yaml_scalar(model.get('target_platform', document.get('target_platform', 'rk3576')))}",
        f"  input_size: [{width}, {height}]",
        "classes:",
    ]
    for idx, item in enumerate(classes):
        if isinstance(item, dict):
            cid = int(item.get("id", idx))
            name = str(item.get("name", f"class_{idx}"))
        else:
            cid = idx
            name = str(item)
        lines.append(f"- id: {cid}")
        lines.append(f"  name: {_yaml_scalar(name)}")
    lines.append("class_names:")
    for name in class_names:
        lines.append(f"- {_yaml_scalar(name)}")
    lines.extend([
        "postprocess:",
        f"  conf_threshold: {float(post.get('conf_threshold', 0.25))}",
        f"  iou_threshold: {float(post.get('iou_threshold', 0.45))}",
        f"  max_det: {int(post.get('max_det', 100))}",
        "runtime:",
        f"  preprocess: {_yaml_scalar(runtime.get('preprocess', 'letterbox'))}",
        f"  color: {_yaml_scalar(runtime.get('color', 'rgb'))}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



class ModelPackageService:
    def __init__(self, model_packages_root: Path, publish_root: Path | None = None) -> None:
        self.model_packages_root = Path(model_packages_root)
        self.publish_root = Path(publish_root) if publish_root else None
        self.model_packages_root.mkdir(parents=True, exist_ok=True)

    def list_packages(self) -> list[dict[str, Any]]:
        if not self.model_packages_root.exists():
            return []
        result = []
        for package_dir in sorted([entry for entry in self.model_packages_root.iterdir() if entry.is_dir()], key=lambda x: x.name):
            summary = self.get_package(package_dir.name, missing_ok=True)
            if summary:
                result.append(summary)
        return result

    def get_package(self, model_id: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        package_dir = self.model_packages_root / _safe_id(model_id)
        if not package_dir.is_dir():
            if missing_ok:
                return None
            raise FileNotFoundError(f"模型包不存在: {model_id}")
        meta = _read_json(package_dir / "package.json", {})
        metrics = _read_json(package_dir / "metrics.json", {})
        yaml_path = package_dir / "model.yaml"
        rknn_path = package_dir / "model.rknn"
        return {
            "model_id": package_dir.name,
            "package_path": str(package_dir),
            "model_name": meta.get("model_name", package_dir.name),
            "version": meta.get("version", "unknown"),
            "task_type": meta.get("task_type", "unknown"),
            "target_platform": meta.get("target_platform", "unknown"),
            "dataset_id": meta.get("dataset_id"),
            "job_id": meta.get("job_id"),
            "status": "ready" if yaml_path.exists() and rknn_path.exists() else "incomplete",
            "has_model_rknn": rknn_path.exists(),
            "has_model_yaml": yaml_path.exists(),
            "metrics": metrics,
            "created_at_ms": meta.get("created_at_ms"),
            "updated_at_ms": meta.get("updated_at_ms"),
        }

    def create_mock_package(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        version: str = "0.1.0",
        task_type: str = "detection",
        dataset_id: str | None = None,
        job_id: str | None = None,
        target_platform: str = "rk3576",
        classes: list[dict[str, Any]] | list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_type = normalize_model_task(task_type)
        safe_model_id = _safe_id(model_id)
        package_dir = self.model_packages_root / safe_model_id
        package_dir.mkdir(parents=True, exist_ok=True)
        name = model_name or safe_model_id
        model_yaml = make_model_yaml(
            model_id=safe_model_id,
            model_name=name,
            version=version,
            task_type=task_type,
            classes=classes,
            target_platform=target_platform,
        )
        (package_dir / "model.rknn").write_bytes(b"VISIONOPS_V3_MOCK_RKNN_PLACEHOLDER\n")
        write_model_yaml(package_dir / "model.yaml", model_yaml)
        now = timestamp_ms()
        package_meta = {
            "schema_version": "1.0",
            "model_id": safe_model_id,
            "model_name": name,
            "version": version,
            "task_type": task_type,
            "target_platform": target_platform,
            "dataset_id": dataset_id,
            "job_id": job_id,
            "created_at_ms": now,
            "updated_at_ms": now,
            "note": "当前为服务端 MVP mock 模型包；真实训练/RKNN 转换接入后会替换 model.rknn。",
        }
        _write_json(package_dir / "package.json", package_meta)
        _write_json(package_dir / "metrics.json", metrics or {"mAP50": None, "source": "mock"})
        _write_json(package_dir / "train_config.yaml.json", train_config or {})
        (package_dir / "export_report.json").write_text(
            json.dumps({"status": "mock", "message": "真实 ONNX/RKNN 导出尚未在本 MVP 中执行。"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (package_dir / "logs").mkdir(exist_ok=True)
        return self.get_package(safe_model_id) or {}

    def publish_package(self, model_id: str, publish_root: Path | None = None) -> dict[str, Any]:
        root = Path(publish_root) if publish_root else self.publish_root
        if root is None:
            raise ValueError("未配置 publish_root，无法发布模型包")
        package_dir = self.model_packages_root / _safe_id(model_id)
        if not package_dir.is_dir():
            raise FileNotFoundError(f"模型包不存在: {model_id}")
        for required in ["model.rknn", "model.yaml"]:
            if not (package_dir / required).is_file():
                raise FileNotFoundError(f"模型包缺少 {required}")
        target_dir = root / package_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_dir / "model.rknn", target_dir / "model.rknn")
        shutil.copy2(package_dir / "model.yaml", target_dir / "model.yaml")
        return {
            "model_id": package_dir.name,
            "publish_path": str(target_dir),
            "files": ["model.rknn", "model.yaml"],
            "published_at_ms": timestamp_ms(),
        }

    def delete_package(self, model_id: str) -> dict[str, Any]:
        package_dir = self.model_packages_root / _safe_id(model_id)
        if not package_dir.is_dir():
            raise FileNotFoundError(f"模型包不存在: {model_id}")
        summary = self.get_package(package_dir.name) or {"model_id": package_dir.name, "package_path": str(package_dir)}
        shutil.rmtree(package_dir)
        summary["deleted"] = True
        summary["deleted_at_ms"] = timestamp_ms()
        return summary


def _safe_id(value: str) -> str:
    value = str(value or "").strip().replace(" ", "_")
    safe = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", "."})
    if not safe or safe in {".", ".."}:
        raise ValueError("非法 ID")
    return safe


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
