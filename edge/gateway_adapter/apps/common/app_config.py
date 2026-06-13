"""业务 App YAML 配置加载与基础校验。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class AppConfigError(ValueError):
    """业务 App 配置无效。"""


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_app_config(path: str | Path | None, defaults: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(defaults))
    if path:
        source = Path(path)
        try:
            loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise AppConfigError(f"无法读取业务配置: {source}: {error}") from error
        if not isinstance(loaded, Mapping):
            raise AppConfigError("业务配置顶层必须是对象")
        config = _merge(config, loaded)

    app = config.get("app")
    rules = config.get("rules")
    if not isinstance(app, Mapping) or not app.get("name"):
        raise AppConfigError("配置缺少 app.name")
    if not isinstance(rules, Mapping):
        raise AppConfigError("配置缺少 rules")
    return config


def business_rules(config: Mapping[str, Any]) -> dict[str, Any]:
    rules = config.get("rules")
    if not isinstance(rules, Mapping):
        raise AppConfigError("配置 rules 必须是对象")
    return dict(rules)
