#!/usr/bin/env python3
"""校验 VisionOps v3 的 edge、task 与 app 分层配置。"""

from __future__ import annotations

import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


SUPPORTED_SCHEMA_VERSION = "1.0"
SUPPORTED_PLATFORMS = {"rk3588", "rk3576"}
SUPPORTED_TASKS = {"detection", "obb", "roi_classification"}
SENSITIVE_KEY_PATTERN = re.compile(r"(?:password|token|secret|private_key)", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(r"^(?:env:[A-Z][A-Z0-9_]*|\$\{[A-Z][A-Z0-9_]*\})$")
PATH_KEY_PATTERN = re.compile(r"(?:_path|_dir|_root)$")


class ConfigValidationError(ValueError):
    """表示配置内容不满足统一配置契约。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """读取单个 YAML 文件，并要求顶层为对象。"""
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError([f"配置文件不存在: {source}"]) from exc
    except yaml.YAMLError as exc:
        raise ConfigValidationError([f"YAML 格式错误 {source}: {exc}"]) from exc

    if not isinstance(data, dict):
        raise ConfigValidationError([f"配置文件顶层必须是对象: {source}"])
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并对象；列表和标量由高优先级配置整体替换。"""
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_edge_config(paths: Sequence[str | Path]) -> dict[str, Any]:
    """按传入顺序合并 edge 基础配置和平台/设备覆盖配置。"""
    if not paths:
        raise ConfigValidationError(["至少需要一个 edge 配置文件"])

    merged: dict[str, Any] = {}
    for index, path in enumerate(paths):
        layer = load_yaml_file(path)
        kind = layer.get("kind")
        if index == 0 and kind != "edge":
            raise ConfigValidationError([f"首个 edge 配置 kind 必须为 edge: {path}"])
        if index > 0 and kind not in {"edge", "edge_overlay"}:
            raise ConfigValidationError([f"edge 覆盖配置 kind 必须为 edge_overlay 或 edge: {path}"])
        merged = deep_merge(merged, layer)

    merged["kind"] = "edge"
    return merged


def load_configuration(
    edge_paths: Sequence[str | Path],
    task_path: str | Path,
    app_paths: Sequence[str | Path],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """加载三类配置，但不执行内容校验。"""
    if not app_paths:
        raise ConfigValidationError(["至少需要一个 app 配置文件"])
    edge = load_edge_config(edge_paths)
    task = load_yaml_file(task_path)
    apps = [load_yaml_file(path) for path in app_paths]
    return edge, task, apps


def _require_mapping(
    data: Mapping[str, Any], key: str, location: str, errors: list[str]
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"缺少必需对象: {location}.{key}")
        return {}
    return value


def _require_nonempty(
    data: Mapping[str, Any], key: str, location: str, errors: list[str]
) -> Any:
    value = data.get(key)
    if value is None or value == "" or value == []:
        errors.append(f"缺少必需字段: {location}.{key}")
    return value


def _validate_header(data: Mapping[str, Any], expected_kind: str, location: str, errors: list[str]) -> None:
    if str(data.get("schema_version", "")) != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            f"{location}.schema_version 必须为 {SUPPORTED_SCHEMA_VERSION}"
        )
    if data.get("kind") != expected_kind:
        errors.append(f"{location}.kind 必须为 {expected_kind}")


def _walk(value: Any, location: str = "config") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield str(key), child_location, child
            yield from _walk(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{location}[{index}]")


def _validate_sensitive_fields(documents: Sequence[tuple[str, Mapping[str, Any]]], errors: list[str]) -> None:
    for name, document in documents:
        for key, location, value in _walk(document, name):
            if not SENSITIVE_KEY_PATTERN.search(key):
                continue
            if not isinstance(value, str) or not REFERENCE_PATTERN.fullmatch(value):
                errors.append(
                    f"敏感字段必须使用 env:NAME 或 ${{NAME}} 引用，禁止写入明文: {location}"
                )


def _validate_paths(documents: Sequence[tuple[str, Mapping[str, Any]]], errors: list[str]) -> None:
    for name, document in documents:
        for key, location, value in _walk(document, name):
            if not PATH_KEY_PATTERN.search(key):
                continue
            if not isinstance(value, str) or not value:
                errors.append(f"路径字段必须是非空字符串: {location}")
                continue
            if value.startswith("env:") or value.startswith("${"):
                continue
            path = PurePosixPath(value)
            if not path.is_absolute():
                errors.append(f"设备运行路径必须是绝对 POSIX 路径: {location}={value}")
            if ".." in path.parts or value.startswith("~"):
                errors.append(f"路径不得包含用户目录缩写或上级跳转: {location}={value}")
            if value == "/home" or value.startswith("/home/"):
                errors.append(f"配置不得依赖开发者主目录: {location}={value}")


def _validate_ports(edge: Mapping[str, Any], apps: Sequence[Mapping[str, Any]], errors: list[str]) -> None:
    occupied: dict[int, str] = {}

    def register(port: Any, location: str) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            errors.append(f"端口必须是 1 到 65535 的整数: {location}")
            return
        previous = occupied.get(port)
        if previous is not None:
            errors.append(f"本地端口冲突: {location} 与 {previous} 均使用 {port}")
            return
        occupied[port] = location

    services = edge.get("services")
    if isinstance(services, Mapping):
        for service_name, service in services.items():
            if not isinstance(service, Mapping):
                errors.append(f"edge.services.{service_name} 必须是对象")
                continue
            for key in ("listen_port", "metrics_port"):
                if key in service:
                    register(service[key], f"edge.services.{service_name}.{key}")

    for app_index, app in enumerate(apps):
        for key, location, value in _walk(app, f"apps[{app_index}]"):
            if key in {"listen_port", "metrics_port"}:
                register(value, location)


def _validate_edge(edge: Mapping[str, Any], errors: list[str]) -> None:
    _validate_header(edge, "edge", "edge", errors)
    _require_mapping(edge, "metadata", "edge", errors)
    device = _require_mapping(edge, "device", "edge", errors)
    camera = _require_mapping(edge, "camera", "edge", errors)
    runtime = _require_mapping(edge, "runtime", "edge", errors)
    services = _require_mapping(edge, "services", "edge", errors)
    paths = _require_mapping(edge, "paths", "edge", errors)

    for key in ("device_id", "platform", "model"):
        _require_nonempty(device, key, "edge.device", errors)
    platform = device.get("platform")
    if platform not in SUPPORTED_PLATFORMS:
        errors.append(
            f"edge.device.platform 不受支持: {platform!r}，允许值为 {sorted(SUPPORTED_PLATFORMS)}"
        )

    for key in ("adapter", "source_ref", "pixel_format", "width", "height", "fps"):
        _require_nonempty(camera, key, "edge.camera", errors)
    source_ref = camera.get("source_ref")
    if isinstance(source_ref, str) and not REFERENCE_PATTERN.fullmatch(source_ref):
        errors.append("edge.camera.source_ref 必须使用环境变量引用")

    if runtime.get("engine") != "cpp_rknn":
        errors.append("edge.runtime.engine 必须为 cpp_rknn")
    for key in ("npu_core", "worker_threads", "max_queue_size"):
        _require_nonempty(runtime, key, "edge.runtime", errors)

    if not services:
        errors.append("edge.services 至少需要一个服务")
    for key in ("install_root", "state_dir", "log_dir", "model_dir"):
        _require_nonempty(paths, key, "edge.paths", errors)


def _validate_task(task: Mapping[str, Any], platform: Any, errors: list[str]) -> None:
    _validate_header(task, "task", "task", errors)
    task_meta = _require_mapping(task, "task", "task", errors)
    model = _require_mapping(task, "model", "task", errors)
    input_config = _require_mapping(task, "input", "task", errors)
    classes = _require_mapping(task, "classes", "task", errors)
    postprocess = _require_mapping(task, "postprocess", "task", errors)
    output = _require_mapping(task, "output", "task", errors)

    for key in ("name", "type", "version"):
        _require_nonempty(task_meta, key, "task.task", errors)
    task_type = task_meta.get("type")
    if task_type not in SUPPORTED_TASKS:
        errors.append(f"task.task.type 不受支持: {task_type!r}")

    _require_nonempty(model, "manifest_ref", "task.model", errors)
    platforms = _require_nonempty(model, "compatible_platforms", "task.model", errors)
    if isinstance(platforms, list):
        invalid = sorted({item for item in platforms if item not in SUPPORTED_PLATFORMS})
        if invalid:
            errors.append(f"task.model.compatible_platforms 包含不支持的平台: {invalid}")
        if platform in SUPPORTED_PLATFORMS and platform not in platforms:
            errors.append(f"任务模型不兼容 edge 平台 {platform}")
    elif platforms is not None:
        errors.append("task.model.compatible_platforms 必须是列表")

    for key in ("width", "height", "color_space", "layout"):
        _require_nonempty(input_config, key, "task.input", errors)
    _require_nonempty(classes, "version", "task.classes", errors)
    names = _require_nonempty(classes, "names", "task.classes", errors)
    if names is not None and (not isinstance(names, list) or not all(isinstance(item, str) and item for item in names)):
        errors.append("task.classes.names 必须是非空字符串列表")
    _require_nonempty(postprocess, "type", "task.postprocess", errors)
    _require_nonempty(output, "schema", "task.output", errors)


def _validate_apps(apps: Sequence[Mapping[str, Any]], errors: list[str]) -> None:
    names: set[str] = set()
    for index, app in enumerate(apps):
        location = f"apps[{index}]"
        _validate_header(app, "app", location, errors)
        app_meta = _require_mapping(app, "app", location, errors)
        name = _require_nonempty(app_meta, "name", f"{location}.app", errors)
        _require_nonempty(app_meta, "version", f"{location}.app", errors)
        if isinstance(name, str):
            if name in names:
                errors.append(f"app 名称重复: {name}")
            names.add(name)


def validate_configuration(
    edge: Mapping[str, Any],
    task: Mapping[str, Any],
    apps: Sequence[Mapping[str, Any]],
) -> None:
    """执行结构、跨配置、安全、路径和端口校验。"""
    errors: list[str] = []
    _validate_edge(edge, errors)
    platform = edge.get("device", {}).get("platform") if isinstance(edge.get("device"), Mapping) else None
    _validate_task(task, platform, errors)
    _validate_apps(apps, errors)

    documents = [("edge", edge), ("task", task)] + [
        (f"apps[{index}]", app) for index, app in enumerate(apps)
    ]
    _validate_sensitive_fields(documents, errors)
    _validate_paths(documents, errors)
    _validate_ports(edge, apps, errors)

    if errors:
        raise ConfigValidationError(errors)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 VisionOps v3 分层配置")
    parser.add_argument("--edge", action="append", required=True, help="edge 配置，可按优先级重复传入")
    parser.add_argument("--task", required=True, help="task 配置")
    parser.add_argument("--app", action="append", required=True, help="app 配置，可重复传入")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        edge, task, apps = load_configuration(args.edge, args.task, args.app)
        validate_configuration(edge, task, apps)
    except ConfigValidationError as exc:
        print("配置校验失败:", file=sys.stderr)
        for error in exc.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("配置校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
