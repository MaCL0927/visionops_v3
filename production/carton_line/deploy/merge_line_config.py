#!/usr/bin/env python3
"""Add newly introduced keys to an installed line YAML while preserving values."""
from __future__ import annotations

import argparse
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def merge(defaults: dict[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(defaults)
    for key, value in current.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), dict):
            output[key] = merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def drop_path(document: dict[str, Any], dotted_path: str) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return
    current: dict[str, Any] = document
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            return
        current = value
    current.pop(parts[-1], None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--drop-path",
        action="append",
        default=[],
        help="remove obsolete dotted key from installed YAML before merge",
    )
    args = parser.parse_args()
    template_path = Path(args.template)
    target_path = Path(args.target)
    defaults = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        raise ValueError("template YAML top-level must be object")
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(defaults, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"created {target_path}")
        return 0
    current = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        raise ValueError("installed YAML top-level must be object")
    original_current = deepcopy(current)
    for dotted_path in args.drop_path:
        drop_path(current, dotted_path)
    merged = merge(defaults, current)
    if merged == original_current:
        print(f"up-to-date {target_path}")
        return 0
    backup = target_path.with_name(f"{target_path.name}.bak.{int(time.time())}")
    shutil.copy2(target_path, backup)
    target_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"merged new keys into {target_path}; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
