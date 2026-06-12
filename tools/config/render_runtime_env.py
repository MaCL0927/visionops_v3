#!/usr/bin/env python3
"""将已校验的 VisionOps v3 分层配置渲染为 runtime env 文本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .validate_config import (
        ConfigValidationError,
        load_configuration,
        validate_configuration,
    )
except ImportError:  # 允许直接执行该脚本。
    from validate_config import (  # type: ignore
        ConfigValidationError,
        load_configuration,
        validate_configuration,
    )


TOOL_VERSION = "1.0"


def _env_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _quote_env(value: str) -> str:
    """使用双引号输出，避免空格、井号和 shell 元字符改变含义。"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _flatten(prefix: str, value: Any, output: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            next_prefix = f"{prefix}_{_env_name(str(key))}" if prefix else _env_name(str(key))
            _flatten(next_prefix, value[key], output)
        return
    output[prefix] = _format_value(value)


def source_digest(paths: Sequence[str | Path]) -> str:
    """根据规范化路径和文件原始内容生成可复现摘要。"""
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item).resolve()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_runtime_env(
    edge: Mapping[str, Any],
    task: Mapping[str, Any],
    apps: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str | Path],
    generated_at: datetime | None = None,
) -> str:
    """渲染 env 文本；调用方负责在写文件前完成原子替换策略。"""
    validate_configuration(edge, task, apps)
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    normalized_paths = [str(Path(path).resolve()) for path in source_paths]
    values: dict[str, str] = {
        "VISIONOPS_CONFIG_SCHEMA_VERSION": str(edge["schema_version"]),
        "VISIONOPS_CONFIG_SOURCE_FILES": ",".join(normalized_paths),
        "VISIONOPS_CONFIG_SOURCE_SHA256": source_digest(source_paths),
        "VISIONOPS_CONFIG_GENERATED_AT": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "VISIONOPS_CONFIG_RENDERER_VERSION": TOOL_VERSION,
    }
    _flatten("VISIONOPS_EDGE", edge, values)
    _flatten("VISIONOPS_TASK", task, values)
    for app in apps:
        app_meta = app.get("app", {})
        app_name = _env_name(str(app_meta.get("name", "APP")))
        _flatten(f"VISIONOPS_APP_{app_name}", app, values)

    lines = [
        "# VisionOps v3 运行时环境配置",
        "# 此文件由配置工具生成，请勿手工修改或提交 Git。",
    ]
    lines.extend(f"{key}={_quote_env(values[key])}" for key in sorted(values))
    return "\n".join(lines) + "\n"


def write_atomic(path: str | Path, content: str) -> None:
    """在目标目录内写临时文件后原子替换。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染 VisionOps v3 runtime env")
    parser.add_argument("--edge", action="append", required=True, help="edge 配置，可按优先级重复传入")
    parser.add_argument("--task", required=True, help="task 配置")
    parser.add_argument("--app", action="append", required=True, help="app 配置，可重复传入")
    parser.add_argument("--output", help="输出文件；不传时写到标准输出")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = [*args.edge, args.task, *args.app]
    try:
        edge, task, apps = load_configuration(args.edge, args.task, args.app)
        content = render_runtime_env(edge, task, apps, source_paths)
    except (ConfigValidationError, OSError) as exc:
        print(f"runtime env 生成失败: {exc}", file=sys.stderr)
        return 1

    if args.output:
        write_atomic(args.output, content)
        print(f"已生成 runtime env: {Path(args.output)}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
