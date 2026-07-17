#!/usr/bin/env python3
"""Launch carton-palletizing stack or segmentation box-grasp profiles."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from production.carton_palletizing.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


def _config_path(value: Optional[str]) -> str:
    return str(Path(value or os.environ.get("VISIONOPS_CARTON_PALLETIZING_CONFIG", DEFAULT_CONFIG_PATH)).expanduser())


def _runtime(config: dict, profile: str) -> int:
    runtime = config["runtime"] if profile == "stack" else config["box_grasp"]["runtime"]
    parsed = urlparse(runtime["url"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("Runtime URL 必须包含明确的 http 端口")
    runtime_bin = Path(os.environ.get(
        "VISIONOPS_RUNTIME_BIN",
        str(PROJECT_ROOT / "build-rknn/edge/runtime_cpp/visionops_runtime_mock"),
    ))
    env_name = "VISIONOPS_CARTON_PALLETIZING_MODEL_DIR" if profile == "stack" else "VISIONOPS_CARTON_BOX_GRASP_MODEL_DIR"
    model_dir = Path(os.environ.get(env_name, runtime["model_dir"]))
    bridge_url = os.environ.get("VISIONOPS_CAMERA_BRIDGE_URL_OVERRIDE", config["camera_bridge"]["base_url"])
    recovery = config["runtime_recovery"]
    if not runtime_bin.is_file() or not os.access(runtime_bin, os.X_OK):
        raise FileNotFoundError("Runtime binary not found or not executable: {}".format(runtime_bin))
    if not (model_dir / "model.rknn").is_file() or not (model_dir / "model.yaml").is_file():
        raise FileNotFoundError("Model package must contain model.rknn and model.yaml: {}".format(model_dir))
    command = [
        str(runtime_bin),
        "--backend", "rknn",
        "--frame-source", "hp60c_bridge",
        "--hp60c-url", bridge_url,
        "--hp60c-snapshot-path", str(config["camera_bridge"]["snapshot_path"]),
        "--hp60c-health-path", str(config["camera_bridge"]["health_path"]),
        "--model-dir", str(model_dir),
        "--roi-config", str(runtime["roi_config_path"]),
        "--preprocess-backend", "auto",
        "--host", parsed.hostname,
        "--port", str(parsed.port),
        "--device-id", str(runtime["device_id"]),
        "--component", str(runtime["component"]),
        "--stale-frame-timeout-ms", str(recovery["stale_frame_timeout_ms"]),
        "--camera-reconnect-failure-threshold", str(recovery["failure_threshold"]),
        "--camera-reconnect-initial-ms", str(recovery["initial_backoff_ms"]),
        "--camera-reconnect-max-ms", str(recovery["max_backoff_ms"]),
    ]
    os.execv(command[0], command)
    return 0


def _app(config_path: str, profile: str) -> int:
    module = (
        "production.carton_palletizing.tasks.first_layer_placement.service"
        if profile == "stack"
        else "production.carton_palletizing.tasks.box_grasp_vision.service"
    )
    command = [sys.executable, "-m", module, "--config", config_path]
    os.execv(command[0], command)
    return 0


def _collector(config: dict, profile: str) -> int:
    if profile == "stack":
        collector = config["collector"]
        runtime = config["runtime"]
        app = config["app"]
        runtime_service = "visionops-v3-carton-palletizing-runtime.service"
        app_service = "visionops-v3-carton-palletizing-app.service"
    else:
        collector = config["box_grasp"]["collector"]
        runtime = config["box_grasp"]["runtime"]
        app = config["box_grasp"]["app"]
        runtime_service = "visionops-v3-carton-box-grasp-runtime.service"
        app_service = "visionops-v3-carton-box-grasp-app.service"
    os.environ["VISIONOPS_COLLECTOR_RUNTIME_SERVICE"] = runtime_service
    os.environ["VISIONOPS_COLLECTOR_CAMERA_DEPENDENT_SERVICES"] = app_service
    app_host = str(app["listen_host"])
    if app_host in {"0.0.0.0", "::"}:
        app_host = "127.0.0.1"
    app_url = "http://{}:{}".format(app_host, app["listen_port"])
    command = [
        sys.executable,
        "-m", "apps.collector_web.backend.main",
        "--host", str(collector["listen_host"]),
        "--port", str(collector["listen_port"]),
        "--runtime-url", str(runtime["url"]),
        "--gateway-url", app_url,
        "--business-app-url", app_url,
        "--production-inference-source", str(collector["production_inference_source"]),
        "--snapshot-refresh-interval-ms", str(collector["snapshot_refresh_interval_ms"]),
        "--status-refresh-interval-ms", str(collector["status_refresh_interval_ms"]),
        "--device-id", str(collector["device_id"]),
        "--component", str(collector["component"]),
        "--models-root", str(collector["models_root"]),
    ]
    os.execv(command[0], command)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps carton-palletizing launcher")
    parser.add_argument("--config", help="默认 production/carton_palletizing/config/line.yaml")
    parser.add_argument(
        "command",
        choices=(
            "runtime", "app", "collector",
            "box-grasp-runtime", "box-grasp-app", "box-grasp-collector",
            "show-config",
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = _config_path(args.config)
    config = load_config(path)
    if args.command == "runtime":
        return _runtime(config, "stack")
    if args.command == "app":
        return _app(path, "stack")
    if args.command == "collector":
        return _collector(config, "stack")
    if args.command == "box-grasp-runtime":
        return _runtime(config, "box_grasp")
    if args.command == "box-grasp-app":
        return _app(path, "box_grasp")
    if args.command == "box-grasp-collector":
        return _collector(config, "box_grasp")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
