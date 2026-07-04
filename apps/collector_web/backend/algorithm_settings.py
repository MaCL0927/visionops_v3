"""Collector Web 算法设置读写。

M15.1 规则：
- 模型包固定为 models/<model_dir>/model.rknn + model.yaml。
- model.yaml 是算法阈值的唯一写入目标。
- 只写当前模型 YAML 中与 Runtime 已支持的公共阈值：score/conf 与 nms。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .model_catalog import find_scanned_model, scan_model_catalog

_SCORE_KEYS = ("score_threshold", "conf_threshold", "confidence_threshold")
_NMS_KEYS = ("nms_threshold", "iou_threshold")
_TASKS_WITH_NMS = {"detection", "obb", "segmentation"}
_TASKS_WITH_SCORE = {"classification", "detection", "obb", "segmentation"}


def get_algorithm_settings_payload(
    models_root: Path,
    *,
    current_model: dict[str, Any] | None = None,
    model_id: str | None = None,
    package_dir: str | None = None,
) -> dict[str, Any]:
    """读取算法设置面板需要的模型列表与选中模型 YAML 阈值。"""

    models = scan_model_catalog(models_root, current_model=current_model)
    selected = _select_model(models, model_id=model_id, package_dir=package_dir)
    if selected is None:
        selected = next((model for model in models if model.get("active") and model.get("valid")), None)
    if selected is None:
        selected = next((model for model in models if model.get("valid")), None)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "message_type": "algorithm_settings",
        "models_root": str(Path(models_root).resolve()),
        "models": models,
        "selected_model": selected,
        "settings": None,
    }
    if selected is None:
        payload["warning"] = "models_root 下没有可用的标准模型包"
        return payload

    yaml_path = Path(str(selected["package_path"])) / "model.yaml"
    document = _load_yaml_document(yaml_path)
    task_type = _task_type(document, selected)
    score_key, score_value = _read_first_number(document, _SCORE_KEYS, 0.5)
    nms_key, nms_value = _read_first_number(document, _NMS_KEYS, 0.45)
    payload["settings"] = {
        "model_id": selected.get("model_id"),
        "package_dir": selected.get("package_dir"),
        "package_path": selected.get("package_path"),
        "yaml_path": str(yaml_path),
        "task_type": task_type,
        "input_size": selected.get("input_size") or _parse_input_size(document.get("input_size")),
        "score_threshold": score_value,
        "score_threshold_key": score_key or "score_threshold",
        "nms_threshold": nms_value if task_type in _TASKS_WITH_NMS else None,
        "nms_threshold_key": nms_key or "nms_threshold",
        "supports_score_threshold": task_type in _TASKS_WITH_SCORE,
        "supports_nms_threshold": task_type in _TASKS_WITH_NMS,
        "active": bool(selected.get("active")),
    }
    return payload


def apply_algorithm_settings(
    models_root: Path,
    payload: dict[str, Any],
    *,
    current_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将算法阈值写入指定模型的 model.yaml。"""

    models = scan_model_catalog(models_root, current_model=current_model)
    selected = _select_model(
        models,
        model_id=str(payload.get("model_id") or "").strip() or None,
        package_dir=str(payload.get("package_dir") or "").strip() or None,
    )
    if selected is None:
        raise ValueError("未找到指定模型包，无法写入算法设置")
    if not selected.get("valid"):
        raise ValueError(f"指定模型包无效: {selected.get('error') or 'unknown'}")

    yaml_path = Path(str(selected["package_path"])) / "model.yaml"
    document = _load_yaml_document(yaml_path)
    task_type = _task_type(document, selected)
    if task_type not in _TASKS_WITH_SCORE:
        raise ValueError(f"暂不支持该任务类型的算法设置: {task_type or 'unknown'}")

    changed = False
    updates: dict[str, Any] = {}
    score = _optional_threshold(payload.get("score_threshold"), "置信度阈值")
    if score is not None:
        score_key = _first_existing_key(document, _SCORE_KEYS) or "score_threshold"
        old_score = _coerce_float(document.get(score_key), None)
        if old_score is None or abs(old_score - score) > 1e-9:
            document[score_key] = float(score)
            updates[score_key] = float(score)
            changed = True

    if task_type in _TASKS_WITH_NMS:
        nms = _optional_threshold(payload.get("nms_threshold"), "NMS 阈值")
        if nms is not None:
            nms_key = _first_existing_key(document, _NMS_KEYS) or "nms_threshold"
            old_nms = _coerce_float(document.get(nms_key), None)
            if old_nms is None or abs(old_nms - nms) > 1e-9:
                document[nms_key] = float(nms)
                updates[nms_key] = float(nms)
                changed = True

    if changed:
        _atomic_write_yaml(yaml_path, document)

    refreshed = get_algorithm_settings_payload(
        models_root,
        current_model=current_model,
        model_id=str(selected.get("model_id") or ""),
    )
    return {
        "schema_version": "1.0",
        "message_type": "algorithm_settings_apply_result",
        "status": "ok",
        "changed": changed,
        "updates": updates,
        "selected_model": refreshed.get("selected_model") or selected,
        "settings": refreshed.get("settings"),
        "models": refreshed.get("models") or models,
        "reload_runtime": bool(changed and selected.get("active")),
    }


def _select_model(
    models: list[dict[str, Any]],
    *,
    model_id: str | None,
    package_dir: str | None,
) -> dict[str, Any] | None:
    if model_id or package_dir:
        return find_scanned_model(models, model_id=model_id, package_dir=package_dir)
    return None


def _load_yaml_document(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取 model.yaml: {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"model.yaml 解析失败: {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("model.yaml 顶层必须是对象")
    return document


def _atomic_write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, allow_unicode=True, sort_keys=False)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _task_type(document: dict[str, Any], selected: dict[str, Any]) -> str:
    return str(document.get("task_type") or document.get("task") or selected.get("task_type") or "").strip().lower()


def _first_existing_key(document: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return next((key for key in keys if key in document), None)


def _read_first_number(document: dict[str, Any], keys: tuple[str, ...], fallback: float) -> tuple[str | None, float]:
    key = _first_existing_key(document, keys)
    if key is None:
        return None, float(fallback)
    return key, _coerce_float(document.get(key), float(fallback))


def _coerce_float(value: Any, fallback: float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number == number else fallback


def _optional_threshold(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}必须是数字") from error
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label}必须在 0~1 之间")
    return number


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
