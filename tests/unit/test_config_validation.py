"""统一配置校验器的单元测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tools.config.validate_config import (
    ConfigValidationError,
    load_configuration,
    validate_configuration,
)
from apps.collector_web.backend.config_loader import load_config as load_collector_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGE_BASE = PROJECT_ROOT / "configs/edge/base.example.yaml"
EDGE_RK3588 = PROJECT_ROOT / "configs/edge/rk3588.example.yaml"
TASK_DETECTION = PROJECT_ROOT / "configs/task/detection.example.yaml"
APP_COLLECTOR = PROJECT_ROOT / "configs/app/collector.example.yaml"
APP_GATEWAY_MODBUS = PROJECT_ROOT / "configs/app/gateway_modbus.example.yaml"


def valid_configuration():
    return load_configuration(
        [EDGE_BASE, EDGE_RK3588],
        TASK_DETECTION,
        [APP_COLLECTOR, APP_GATEWAY_MODBUS],
    )


def test_valid_configuration_passes() -> None:
    edge, task, apps = valid_configuration()
    validate_configuration(edge, task, apps)


def test_missing_required_field_fails() -> None:
    edge, task, apps = valid_configuration()
    invalid_edge = deepcopy(edge)
    del invalid_edge["device"]["device_id"]

    with pytest.raises(ConfigValidationError, match="device_id"):
        validate_configuration(invalid_edge, task, apps)


def test_plaintext_sensitive_field_fails() -> None:
    edge, task, apps = valid_configuration()
    invalid_apps = deepcopy(apps)
    invalid_apps[1]["gateway"]["password"] = "example-plaintext"

    with pytest.raises(ConfigValidationError, match="敏感字段"):
        validate_configuration(edge, task, invalid_apps)


def test_port_conflict_fails() -> None:
    edge, task, apps = valid_configuration()
    invalid_edge = deepcopy(edge)
    invalid_edge["services"]["collector_web"]["listen_port"] = 8082

    with pytest.raises(ConfigValidationError, match="端口冲突"):
        validate_configuration(invalid_edge, task, apps)


def test_collector_config_loads_m7_downstreams() -> None:
    config = load_collector_config([
        "--config", str(APP_COLLECTOR),
        "--port", "18090",
    ])
    assert config.port == 18090
    assert config.runtime_url == "http://127.0.0.1:18080"
    assert config.gateway_url == "http://127.0.0.1:19090"
    assert config.business_app_url == "http://127.0.0.1:19110"
    assert config.snapshot_refresh_interval_ms == 1000
    assert config.status_refresh_interval_ms == 2000
