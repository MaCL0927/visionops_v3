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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--target", required=True)
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
    merged = merge(defaults, current)
    if merged == current:
        print(f"up-to-date {target_path}")
        return 0
    backup = target_path.with_name(f"{target_path.name}.bak.{int(time.time())}")
    shutil.copy2(target_path, backup)
    target_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"merged new keys into {target_path}; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
