#!/usr/bin/env python3
"""输出 VisionOps v3 当前接口契约和示例摘要。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "interfaces/schemas"
DEFAULT_EXAMPLE_DIR = PROJECT_ROOT / "interfaces/examples"


def build_summary(schema_dir: str | Path, example_dir: str | Path) -> dict[str, object]:
    """读取文件列表并统计示例中的消息和任务类型。"""
    schema_root = Path(schema_dir)
    example_root = Path(example_dir)
    schema_files = sorted(path.name for path in schema_root.glob("*.schema.json"))
    example_files = sorted(example_root.glob("*.example.json"))

    message_types: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    examples: list[str] = []
    for path in example_files:
        document = json.loads(path.read_text(encoding="utf-8"))
        examples.append(path.name)
        message_type = document.get("message_type")
        task_type = document.get("task_type")
        if isinstance(message_type, str):
            message_types[message_type] += 1
        if isinstance(task_type, str):
            task_types[task_type] += 1

    return {
        "schema_files": schema_files,
        "example_files": examples,
        "message_types": dict(sorted(message_types.items())),
        "task_types": dict(sorted(task_types.items())),
    }


def print_text_summary(summary: dict[str, object]) -> None:
    """以便于终端阅读的中文格式输出摘要。"""
    print("接口 Schema 文件:")
    for name in summary["schema_files"]:  # type: ignore[union-attr]
        print(f"- {name}")
    print("\n示例 JSON 文件:")
    for name in summary["example_files"]:  # type: ignore[union-attr]
        print(f"- {name}")
    print("\nmessage_type 摘要:")
    for name, count in summary["message_types"].items():  # type: ignore[union-attr]
        print(f"- {name}: {count}")
    print("\ntask_type 摘要:")
    for name, count in summary["task_types"].items():  # type: ignore[union-attr]
        print(f"- {name}: {count}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出接口契约与示例摘要")
    parser.add_argument("--schema-dir", default=str(DEFAULT_SCHEMA_DIR), help="schema 目录")
    parser.add_argument("--example-dir", default=str(DEFAULT_EXAMPLE_DIR), help="example 目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args.schema_dir, args.example_dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_text_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
