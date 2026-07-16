"""Collector Web 的 YAML 与命令行配置加载。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import yaml

from .model_catalog import default_models_root


@dataclass(frozen=True)
class CollectorConfig:
    """Collector Web 进程启动配置。"""

    host: str = "0.0.0.0"
    port: int = 8090
    runtime_url: str = "http://127.0.0.1:18080"
    gateway_url: str = "http://127.0.0.1:19090"
    business_app_url: str = "http://127.0.0.1:19110"
    snapshot_refresh_interval_ms: int = 200
    status_refresh_interval_ms: int = 2000
    device_id: str = "example-edge-001"
    component: str = "collector_web"
    models_root: str = ""
    production_inference_source: str = "runtime"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须位于 1 到 65535")
    return port


def _service_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("服务 URL 必须是有效的 HTTP 或 HTTPS URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("服务 URL 只能包含协议、主机和可选端口")
    return normalized


def _positive_ms(value: str) -> int:
    number = int(value)
    if number < 100:
        raise argparse.ArgumentTypeError("刷新间隔不得小于 100 ms")
    return number


def _path_text(value: str) -> str:
    return str(Path(value).expanduser())


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 Collector 配置: {source}: {error}") from error
    if not isinstance(document, dict) or document.get("kind") != "app":
        raise ValueError("Collector 配置必须是 kind=app 的 YAML 对象")
    collector = document.get("collector", {})
    if not isinstance(collector, dict):
        raise ValueError("collector 配置必须是对象")
    return collector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps v3 Collector Web 后端")
    parser.add_argument("--config", help="Collector app YAML 配置")
    parser.add_argument("--host", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=_port, help="监听端口，默认 8090")
    parser.add_argument(
        "--runtime-url",
        type=_service_url,
        help="Runtime HTTP 地址",
    )
    parser.add_argument("--gateway-url", type=_service_url, help="Gateway HTTP 地址")
    parser.add_argument("--business-app-url", type=_service_url, help="Business App HTTP 地址")
    parser.add_argument("--snapshot-refresh-interval-ms", type=_positive_ms)
    parser.add_argument("--status-refresh-interval-ms", type=_positive_ms)
    parser.add_argument("--device-id", help="设备标识")
    parser.add_argument("--component", help="组件名称")
    parser.add_argument("--models-root", type=_path_text, help="模型包根目录")
    parser.add_argument(
        "--production-inference-source",
        choices=("runtime", "app"),
        help="生产画面推理来源：runtime 直接推理，app 由业务应用推理并返回可视化结果",
    )
    return parser


def load_config(argv: Sequence[str] | None = None) -> CollectorConfig:
    args = build_parser().parse_args(argv)
    try:
        yaml_config = _load_yaml(args.config)
    except ValueError as error:
        build_parser().error(str(error))

    service = yaml_config.get("service", {}) if isinstance(yaml_config.get("service"), dict) else {}
    downstream = yaml_config.get("downstream", {}) if isinstance(yaml_config.get("downstream"), dict) else {}
    refresh = yaml_config.get("refresh", {}) if isinstance(yaml_config.get("refresh"), dict) else {}
    models = yaml_config.get("models", {}) if isinstance(yaml_config.get("models"), dict) else {}
    production = yaml_config.get("production", {}) if isinstance(yaml_config.get("production"), dict) else {}
    project_root = Path(__file__).resolve().parents[3]
    values = {
        "host": args.host or service.get("listen_host") or "0.0.0.0",
        "port": args.port or service.get("listen_port") or 8090,
        "runtime_url": args.runtime_url or downstream.get("runtime_url") or "http://127.0.0.1:18080",
        "gateway_url": args.gateway_url or downstream.get("gateway_url") or "http://127.0.0.1:19090",
        "business_app_url": args.business_app_url or downstream.get("business_app_url") or "http://127.0.0.1:19110",
        "snapshot_refresh_interval_ms": args.snapshot_refresh_interval_ms or refresh.get("snapshot_refresh_interval_ms") or 200,
        "status_refresh_interval_ms": args.status_refresh_interval_ms or refresh.get("status_refresh_interval_ms") or 2000,
        "device_id": args.device_id or yaml_config.get("device_id") or "example-edge-001",
        "component": args.component or yaml_config.get("component") or "collector_web",
        "models_root": args.models_root or models.get("models_root") or str(default_models_root(project_root)),
        "production_inference_source": (
            args.production_inference_source
            or production.get("inference_source")
            or "runtime"
        ),
    }
    if values["production_inference_source"] not in {"runtime", "app"}:
        build_parser().error("production_inference_source 必须为 runtime 或 app")
    if not values["host"] or not values["device_id"] or not values["component"]:
        build_parser().error("host、device-id 和 component 不能为空")
    for key in ("port", "snapshot_refresh_interval_ms", "status_refresh_interval_ms"):
        if isinstance(values[key], bool) or not isinstance(values[key], int):
            build_parser().error(f"{key} 必须是整数")
    if not 1 <= values["port"] <= 65535:
        build_parser().error("port 必须位于 1..65535")
    if values["snapshot_refresh_interval_ms"] < 100 or values["status_refresh_interval_ms"] < 100:
        build_parser().error("刷新间隔不得小于 100 ms")
    for key in ("runtime_url", "gateway_url", "business_app_url"):
        try:
            values[key] = _service_url(str(values[key]))
        except argparse.ArgumentTypeError as error:
            build_parser().error(f"{key}: {error}")
    return CollectorConfig(**values)
