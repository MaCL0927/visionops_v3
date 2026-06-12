#!/usr/bin/env python3
"""使用 Python 标准库对接口 schema 和示例执行轻量校验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "interfaces/schemas"
DEFAULT_EXAMPLE_DIR = PROJECT_ROOT / "interfaces/examples"

COMMON_REQUIRED = {
    "schema_version",
    "message_type",
    "device_id",
    "component",
    "timestamp_ms",
    "trace_id",
    "source",
    "status",
}

MESSAGE_REQUIRED = {
    "camera_frame": {
        "frame_id",
        "camera_id",
        "width",
        "height",
        "pixel_format",
        "transport",
        "sequence",
        "dropped_count",
    },
    "inference_result": {
        "frame_id",
        "result_id",
        "task_type",
        "model",
        "image",
        "timing",
    },
    "runtime_status": {
        "running",
        "mode",
        "health",
        "uptime_s",
        "loaded_model",
        "camera_connected",
        "fps",
        "latency_ms",
        "counters",
        "last_result_id",
        "last_frame_id",
        "last_error",
    },
    "model_package_manifest": {
        "package_id",
        "model_name",
        "model_version",
        "task_type",
        "created_at",
        "target_platform",
        "files",
        "input",
        "output",
        "postprocess",
        "compatibility",
        "notes",
    },
    "gateway_message": {
        "frame_id",
        "message_id",
        "result_id",
        "app_id",
        "protocol",
        "sequence",
        "heartbeat",
        "final_code",
        "final_label",
        "ok",
        "reason",
    },
}

SUPPORTED_TASK_TYPES = {
    "detection",
    "obb",
    "segmentation",
    "roi_classification",
    "classification",
}

FORBIDDEN_DEBUG_KEYS = {
    "image",
    "image_path",
    "raw_image",
    "raw_tensor",
    "tensor_data",
    "base64",
}


class InterfaceValidationError(ValueError):
    """表示一个或多个接口文件未通过轻量校验。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def load_json_object(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件并要求顶层为对象。"""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InterfaceValidationError([f"文件不存在: {source}"]) from exc
    except json.JSONDecodeError as exc:
        raise InterfaceValidationError(
            [f"JSON 解析失败 {source}:{exc.lineno}:{exc.colno}: {exc.msg}"]
        ) from exc

    if not isinstance(value, dict):
        raise InterfaceValidationError([f"JSON 顶层必须是对象: {source}"])
    return value


def _missing_fields(document: Mapping[str, Any], required: set[str]) -> list[str]:
    return sorted(key for key in required if key not in document)


def _validate_error(error: Any, location: str, errors: list[str]) -> None:
    if error is None:
        return
    if not isinstance(error, Mapping):
        errors.append(f"{location}.error 必须是对象或 null")
        return
    missing = _missing_fields(error, {"code", "message", "detail", "recoverable"})
    if missing:
        errors.append(f"{location}.error 缺少字段: {', '.join(missing)}")


def _validate_inference(document: Mapping[str, Any], location: str, errors: list[str]) -> None:
    task_type = document.get("task_type")
    if task_type not in SUPPORTED_TASK_TYPES:
        errors.append(f"{location}.task_type 不受支持: {task_type!r}")

    model = document.get("model")
    if not isinstance(model, Mapping):
        errors.append(f"{location}.model 必须是对象")
    else:
        missing = _missing_fields(
            model,
            {"model_id", "model_name", "model_version", "backend", "input_size"},
        )
        if missing:
            errors.append(f"{location}.model 缺少字段: {', '.join(missing)}")

    timing = document.get("timing")
    if not isinstance(timing, Mapping):
        errors.append(f"{location}.timing 必须是对象")
    else:
        missing = _missing_fields(
            timing,
            {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"},
        )
        if missing:
            errors.append(f"{location}.timing 缺少字段: {', '.join(missing)}")

    debug = document.get("debug")
    if isinstance(debug, Mapping):
        forbidden = sorted(FORBIDDEN_DEBUG_KEYS.intersection(debug))
        if forbidden:
            errors.append(f"{location}.debug 禁止包含大数据或图片字段: {', '.join(forbidden)}")


def validate_example(document: Mapping[str, Any], location: str = "example") -> None:
    """校验单个示例的公共字段和消息类型关键字段。"""
    errors: list[str] = []
    missing_common = _missing_fields(document, COMMON_REQUIRED)
    if missing_common:
        errors.append(f"{location} 缺少公共字段: {', '.join(missing_common)}")

    schema_version = document.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        errors.append(f"{location}.schema_version 必须是非空字符串")
    message_type = document.get("message_type")
    if message_type not in MESSAGE_REQUIRED:
        errors.append(f"{location}.message_type 不受支持: {message_type!r}")
    else:
        missing = _missing_fields(document, MESSAGE_REQUIRED[message_type])
        if missing:
            errors.append(f"{location} 缺少 {message_type} 字段: {', '.join(missing)}")

    timestamp_ms = document.get("timestamp_ms")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        errors.append(f"{location}.timestamp_ms 必须是非负整数")
    if document.get("status") not in {"ok", "degraded", "error"}:
        errors.append(f"{location}.status 必须为 ok、degraded 或 error")
    _validate_error(document.get("error"), location, errors)

    if message_type == "inference_result":
        _validate_inference(document, location, errors)

    if errors:
        raise InterfaceValidationError(errors)


def validate_directories(
    schema_dir: str | Path = DEFAULT_SCHEMA_DIR,
    example_dir: str | Path = DEFAULT_EXAMPLE_DIR,
) -> list[Path]:
    """解析全部 schema，并校验全部 example，返回已验证示例路径。"""
    schema_root = Path(schema_dir)
    example_root = Path(example_dir)
    errors: list[str] = []

    schema_files = sorted(schema_root.glob("*.schema.json"))
    if not schema_files:
        errors.append(f"未找到 schema 文件: {schema_root}")
    for path in schema_files:
        try:
            schema = load_json_object(path)
            if "$schema" not in schema or "$id" not in schema:
                errors.append(f"schema 缺少 $schema 或 $id: {path}")
        except InterfaceValidationError as exc:
            errors.extend(exc.errors)

    example_files = sorted(example_root.glob("*.example.json"))
    if not example_files:
        errors.append(f"未找到 example 文件: {example_root}")
    for path in example_files:
        try:
            validate_example(load_json_object(path), str(path))
        except InterfaceValidationError as exc:
            errors.extend(exc.errors)

    if errors:
        raise InterfaceValidationError(errors)
    return example_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轻量校验接口 schema 与 example JSON")
    parser.add_argument("--schema-dir", default=str(DEFAULT_SCHEMA_DIR), help="schema 目录")
    parser.add_argument("--example-dir", default=str(DEFAULT_EXAMPLE_DIR), help="example 目录")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        examples = validate_directories(args.schema_dir, args.example_dir)
    except InterfaceValidationError as exc:
        print("接口示例校验失败:", file=sys.stderr)
        for error in exc.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"接口示例校验通过: {len(examples)} 个示例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
