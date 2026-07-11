#!/usr/bin/env python3
"""Start carton-line Runtime, Collector or Gateway from one line YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from production.carton_line.gateway.config import PROJECT_ROOT, load_config


DEFAULT_CONFIG = PROJECT_ROOT / "production/carton_line/config/line.yaml"


def _config_path(value: str | None) -> str:
    return str(Path(value or os.environ.get("VISIONOPS_CARTON_LINE_CONFIG", DEFAULT_CONFIG)).expanduser())


def _task(value: str) -> str:
    if value not in {"partition", "tube", "pick"}:
        raise argparse.ArgumentTypeError("task 必须是 partition、tube 或 pick")
    return value


def _runtime(task: str, config: dict) -> int:
    runtime = config["runtimes"][task]
    parsed = urlparse(runtime["url"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError(f"runtimes.{task}.url 必须包含明确的 http 端口")

    runtime_bin = Path(
        os.environ.get(
            "VISIONOPS_RUNTIME_BIN",
            str(PROJECT_ROOT / "build-rknn/edge/runtime_cpp/visionops_runtime_mock"),
        )
    )
    model_override = os.environ.get(f"VISIONOPS_{task.upper()}_MODEL_DIR")
    model_dir = Path(model_override or runtime["model_dir"])
    bridge_url = os.environ.get("VISIONOPS_CAMERA_BRIDGE_URL", config["camera_bridge"]["base_url"])

    if not runtime_bin.is_file() or not os.access(runtime_bin, os.X_OK):
        raise FileNotFoundError(f"Runtime binary not found or not executable: {runtime_bin}")
    if not (model_dir / "model.rknn").is_file() or not (model_dir / "model.yaml").is_file():
        raise FileNotFoundError(f"Model package must contain model.rknn and model.yaml: {model_dir}")

    command = [
        str(runtime_bin),
        "--backend", "rknn",
        "--frame-source", "hp60c_bridge",
        "--hp60c-url", bridge_url,
        "--hp60c-snapshot-path", str(config["camera_bridge"]["snapshot_path"]),
        "--hp60c-health-path", str(config["camera_bridge"]["health_path"]),
        "--model-dir", str(model_dir),
        "--preprocess-backend", "auto",
        "--host", parsed.hostname,
        "--port", str(parsed.port),
        "--device-id", str(runtime["device_id"]),
        "--component", str(runtime["component"]),
    ]
    os.execv(command[0], command)
    return 0


def _collector(task: str, config: dict) -> int:
    collector = config["collectors"][task]
    runtime_url = config["runtimes"][task]["url"]
    gateway_host = str(config["service"]["listen_host"])
    if gateway_host in {"0.0.0.0", "::"}:
        gateway_host = "127.0.0.1"
    if task == "pick":
        pick_http = config["pick"]["tcp"]["http"]
        pick_host = str(pick_http["listen_host"])
        if pick_host in {"0.0.0.0", "::"}:
            pick_host = "127.0.0.1"
        gateway_url = f"http://{pick_host}:{pick_http['listen_port']}"
        business_url = gateway_url
    else:
        gateway_url = f"http://{gateway_host}:{config['service']['listen_port']}"
        app_port = config["service"]["partition_app_port" if task == "partition" else "tube_app_port"]
        business_url = f"http://{gateway_host}:{app_port}"

    command = [
        sys.executable,
        "-m", "apps.collector_web.backend.main",
        "--host", str(collector["listen_host"]),
        "--port", str(collector["listen_port"]),
        "--runtime-url", runtime_url,
        "--gateway-url", gateway_url,
        "--business-app-url", business_url,
        "--snapshot-refresh-interval-ms", str(collector["snapshot_refresh_interval_ms"]),
        "--status-refresh-interval-ms", str(collector["status_refresh_interval_ms"]),
        "--device-id", str(collector["device_id"]),
        "--component", str(collector["component"]),
        "--models-root", str(collector["models_root"]),
    ]
    os.execv(command[0], command)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps carton-line launcher")
    parser.add_argument("--config", help="产线统一 YAML，默认 production/carton_line/config/line.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime = subparsers.add_parser("runtime", help="启动某个任务的 RKNN Runtime")
    runtime.add_argument("task", type=_task)

    collector = subparsers.add_parser("collector", help="启动某个任务的 Collector Web")
    collector.add_argument("task", type=_task)

    subparsers.add_parser("gateway", help="启动统一 Robot Protocol Gateway")
    subparsers.add_parser("tcp-pick", help="启动纸筒抓取点 TCP Client 服务")
    subparsers.add_parser("show-config", help="输出解析后的统一配置")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = _config_path(args.config)
    config = load_config(path)
    if args.command == "runtime":
        return _runtime(args.task, config)
    if args.command == "collector":
        return _collector(args.task, config)
    if args.command == "gateway":
        from production.carton_line.gateway.service import main as gateway_main
        return gateway_main(["--config", path])
    if args.command == "tcp-pick":
        from production.carton_line.tasks.tube_pick_vision.service import main as pick_main
        return pick_main(["--config", path])
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
