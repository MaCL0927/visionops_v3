"""Collector Web 命令行配置加载。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlparse


@dataclass(frozen=True)
class CollectorConfig:
    """Collector Web 进程启动配置。"""

    host: str = "0.0.0.0"
    port: int = 8090
    runtime_url: str = "http://127.0.0.1:18080"
    device_id: str = "example-edge-001"
    component: str = "collector_web"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须位于 1 到 65535")
    return port


def _runtime_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("runtime-url 必须是有效的 HTTP 或 HTTPS URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("runtime-url 只能包含协议、主机和可选端口")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps v3 Collector Web 后端")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=_port, default=8090, help="监听端口，默认 8090")
    parser.add_argument(
        "--runtime-url",
        type=_runtime_url,
        default="http://127.0.0.1:18080",
        help="Runtime HTTP 地址",
    )
    parser.add_argument("--device-id", default="example-edge-001", help="设备标识")
    parser.add_argument("--component", default="collector_web", help="组件名称")
    return parser


def load_config(argv: Sequence[str] | None = None) -> CollectorConfig:
    args = build_parser().parse_args(argv)
    if not args.host or not args.device_id or not args.component:
        build_parser().error("host、device-id 和 component 不能为空")
    return CollectorConfig(
        host=args.host,
        port=args.port,
        runtime_url=args.runtime_url,
        device_id=args.device_id,
        component=args.component,
    )
