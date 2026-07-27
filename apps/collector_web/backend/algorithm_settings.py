"""Collector Web 算法设置读写。

算法设置规则：
- 模型包固定为 models/<model_dir>/model.rknn + model.yaml。
- model.yaml 是算法阈值的唯一写入目标。
- 只写当前模型 YAML 中与 Runtime 已支持的公共阈值：score/conf 与 nms。
- 保存时只原位修改阈值标量，不重新序列化整个 YAML，避免破坏内联数组、
  数字字符串以及 preprocess.input_roi 等部署字段的原始格式。
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml

from .model_catalog import find_scanned_model, scan_model_catalog

_SCORE_KEYS = ("score_threshold", "conf_threshold", "confidence_threshold")
_NMS_KEYS = ("nms_threshold", "iou_threshold")
_TASKS_WITH_NMS = {"detection", "obb", "segmentation"}
_TASKS_WITH_SCORE = {"classification", "detection", "obb", "segmentation"}
_POSTPROCESS_SECTION = "postprocess"
_CANONICAL_SCORE_KEY = "conf_threshold"
_CANONICAL_NMS_KEY = "iou_threshold"
_MAPPING_KEY_RE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:(?P<rest>.*)$")


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
    score_path, score_value = _read_threshold(document, _SCORE_KEYS, 0.5)
    nms_path, nms_value = _read_threshold(document, _NMS_KEYS, 0.45)
    payload["settings"] = {
        "model_id": selected.get("model_id"),
        "package_dir": selected.get("package_dir"),
        "package_path": selected.get("package_path"),
        "yaml_path": str(yaml_path),
        "task_type": task_type,
        "input_size": selected.get("input_size") or _parse_input_size(document.get("input_size")),
        "score_threshold": score_value,
        "score_threshold_key": ".".join(score_path) if score_path else _CANONICAL_SCORE_KEY,
        "nms_threshold": nms_value if task_type in _TASKS_WITH_NMS else None,
        "nms_threshold_key": ".".join(nms_path) if nms_path else _CANONICAL_NMS_KEY,
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
    """将算法阈值原位写入指定模型的 model.yaml。

    不再使用 ``yaml.safe_dump`` 重写整个文档。这样可以保证：
    - ``input_size: [640, 640]`` 不会变成块列表；
    - ``pixel_xyxy`` / ``normalized_xyxy`` 保持内联数组；
    - ``20260725_165208`` 这类 YAML 1.1 数字样式字符串不会丢失下划线；
    - 注释、字段顺序和其他任务专用字段不变。
    """

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
    original_text = _read_yaml_text(yaml_path)
    document = _load_yaml_document_from_text(original_text, yaml_path)
    task_type = _task_type(document, selected)
    if task_type not in _TASKS_WITH_SCORE:
        raise ValueError(f"暂不支持该任务类型的算法设置: {task_type or 'unknown'}")

    requested_updates: list[tuple[tuple[str, ...], float]] = []
    response_updates: dict[str, Any] = {}

    score = _optional_threshold(payload.get("score_threshold"), "置信度阈值")
    if score is not None:
        score_paths = _existing_threshold_paths(document, _SCORE_KEYS)
        if not score_paths:
            score_paths = [(_POSTPROCESS_SECTION, _CANONICAL_SCORE_KEY)]
        for path in score_paths:
            requested_updates.append((path, score))
            response_updates[".".join(path)] = score

    if task_type in _TASKS_WITH_NMS:
        nms = _optional_threshold(payload.get("nms_threshold"), "NMS 阈值")
        if nms is not None:
            nms_paths = _existing_threshold_paths(document, _NMS_KEYS)
            if not nms_paths:
                nms_paths = [(_POSTPROCESS_SECTION, _CANONICAL_NMS_KEY)]
            for path in nms_paths:
                requested_updates.append((path, nms))
                response_updates[".".join(path)] = nms

    updated_text, changed = _patch_yaml_scalars(original_text, requested_updates)
    if changed:
        # 写入前再用 PyYAML 做语义校验，但绝不使用其输出覆盖原文件。
        _load_yaml_document_from_text(updated_text, yaml_path)
        _atomic_write_text(yaml_path, updated_text)

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
        "updates": response_updates if changed else {},
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


def _read_yaml_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"无法读取 model.yaml: {path}: {error}") from error


def _load_yaml_document(path: Path) -> dict[str, Any]:
    return _load_yaml_document_from_text(_read_yaml_text(path), path)


def _load_yaml_document_from_text(text: str, path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"model.yaml 解析失败: {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("model.yaml 顶层必须是对象")
    return document


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        original_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        original_mode = 0o644

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, original_mode)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _task_type(document: dict[str, Any], selected: dict[str, Any]) -> str:
    return str(document.get("task_type") or document.get("task") or selected.get("task_type") or "").strip().lower()


def _postprocess(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get(_POSTPROCESS_SECTION)
    return value if isinstance(value, dict) else {}


def _existing_threshold_paths(document: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    postprocess = _postprocess(document)
    for key in keys:
        if key in postprocess:
            paths.append((_POSTPROCESS_SECTION, key))
    for key in keys:
        if key in document:
            paths.append((key,))
    return paths


def _read_threshold(
    document: dict[str, Any],
    keys: tuple[str, ...],
    fallback: float,
) -> tuple[tuple[str, ...] | None, float]:
    paths = _existing_threshold_paths(document, keys)
    if not paths:
        return None, float(fallback)
    path = paths[0]
    value: Any = document
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return path, _coerce_float(value, float(fallback))


def _mapping_entries(lines: list[str]) -> list[tuple[tuple[str, ...], int, int]]:
    """返回 ``(path, line_index, indent)``，只识别普通 YAML mapping key。"""

    entries: list[tuple[tuple[str, ...], int, int]] = []
    stack: list[tuple[int, str]] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r\n")
        match = _MAPPING_KEY_RE.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        key = match.group("key")
        rest = match.group("rest")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        path = tuple(item[1] for item in stack) + (key,)
        entries.append((path, index, indent))
        value_without_comment = rest.split("#", 1)[0].strip()
        if not value_without_comment:
            stack.append((indent, key))
    return entries


def _patch_yaml_scalars(
    text: str,
    updates: Iterable[tuple[tuple[str, ...], float]],
) -> tuple[str, bool]:
    """原位替换/插入阈值标量，保持其余字节级文本不变。"""

    update_map: dict[tuple[str, ...], float] = {}
    for path, value in updates:
        update_map[tuple(path)] = float(value)
    if not update_map:
        return text, False

    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    changed = False

    # 先替换已存在的路径。每次重新扫描，保证前序插入不会令索引失效。
    missing = dict(update_map)
    entries = _mapping_entries(lines)
    entry_map: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for path, index, indent in entries:
        entry_map.setdefault(path, []).append((index, indent))

    for path, value in list(missing.items()):
        locations = entry_map.get(path, [])
        if not locations:
            continue
        for index, _indent in locations:
            new_line = _replace_scalar_line(lines[index], path[-1], value)
            if new_line != lines[index]:
                lines[index] = new_line
                changed = True
        missing.pop(path, None)

    # 当前设置只会新增 postprocess 下的规范阈值；按父 mapping 原位插入。
    for path, value in missing.items():
        if len(path) != 2 or path[0] != _POSTPROCESS_SECTION:
            raise ValueError(f"无法在 model.yaml 中新增字段: {'.'.join(path)}")
        lines, inserted = _insert_mapping_scalar(lines, (_POSTPROCESS_SECTION,), path[-1], value)
        changed = changed or inserted

    return "".join(lines), changed


def _replace_scalar_line(raw_line: str, key: str, value: float) -> str:
    newline = ""
    body = raw_line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"

    match = re.match(
        rf"^(?P<prefix>\s*{re.escape(key)}\s*:\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$",
        body,
    )
    if match is None:
        raise ValueError(f"无法原位更新 model.yaml 字段: {key}")
    scalar = _format_threshold(value)
    comment = match.group("comment") or ""
    return f"{match.group('prefix')}{scalar}{comment}{newline}"


def _insert_mapping_scalar(
    lines: list[str],
    parent_path: tuple[str, ...],
    key: str,
    value: float,
) -> tuple[list[str], bool]:
    entries = _mapping_entries(lines)
    parent = next(((index, indent) for path, index, indent in entries if path == parent_path), None)
    newline = _preferred_newline(lines)
    scalar = _format_threshold(value)

    if parent is None:
        if lines and lines[-1] and not lines[-1].endswith(("\n", "\r\n")):
            lines[-1] += newline
        lines.extend([
            f"{_POSTPROCESS_SECTION}:{newline}",
            f"  {key}: {scalar}{newline}",
        ])
        return lines, True

    parent_index, parent_indent = parent
    block_end = len(lines)
    child_indent = parent_indent + 2
    for path, index, indent in entries:
        if index <= parent_index:
            continue
        if indent <= parent_indent:
            block_end = index
            break
        if path[: len(parent_path)] == parent_path and len(path) == len(parent_path) + 1:
            child_indent = indent

    lines.insert(block_end, f"{' ' * child_indent}{key}: {scalar}{newline}")
    return lines, True


def _preferred_newline(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def _format_threshold(value: float) -> str:
    return format(float(value), ".12g")


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
