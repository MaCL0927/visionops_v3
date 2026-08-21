#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch Runtime, M41 App or Collector for carton_bundle_grasp."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from production.carton_bundle_grasp.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


def _config_path(value: Optional[str]) -> str:
    return str(Path(value or os.environ.get("VISIONOPS_CARTON_BUNDLE_GRASP_CONFIG", DEFAULT_CONFIG_PATH)).expanduser())


def _runtime(config: dict) -> int:
    task = config["carton_bundle_grasp"]
    runtime = task["runtime"]
    parsed = urlparse(runtime["url"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("runtime.url 必须包含明确的 http 端口")
    runtime_bin = Path(os.environ.get(
        "VISIONOPS_RUNTIME_BIN",
        str(PROJECT_ROOT / "build-rknn/edge/runtime_cpp/visionops_runtime_mock"),
    ))
    model_dir = Path(os.environ.get("VISIONOPS_CARTON_BUNDLE_GRASP_MODEL_DIR", runtime["model_dir"]))
    if not runtime_bin.is_file() or not os.access(runtime_bin, os.X_OK):
        raise FileNotFoundError("Runtime binary not found or not executable: {}".format(runtime_bin))
    if not (model_dir / "model.rknn").is_file() or not (model_dir / "model.yaml").is_file():
        raise FileNotFoundError("Model package must contain model.rknn and model.yaml: {}".format(model_dir))

    bridge = config["camera_bridge"]
    bridge_url = os.environ.get("VISIONOPS_CAMERA_BRIDGE_URL_OVERRIDE", bridge["base_url"])
    active_model = str(config.get("active_camera", {}).get("camera_model") or bridge.get("camera_model") or "orbbec336l")
    use_shared_rgb = bool(bridge.get("shared_rgb_enabled", True)) and active_model == "orbbec336l"
    frame_source = "shared_memory" if use_shared_rgb else "hp60c_bridge"
    recovery = config["runtime_recovery"]
    algorithm = task["algorithm"]
    command = [
        str(runtime_bin),
        "--backend", "rknn",
        "--frame-source", frame_source,
        "--hp60c-url", str(bridge_url),
        "--hp60c-snapshot-path", str(bridge["snapshot_path"]),
        "--hp60c-health-path", str(bridge["health_path"]),
        "--shared-memory-name", str(bridge.get("shared_rgb_name", "/visionops_orbbec336l_rgb")),
        "--shared-memory-fallback-http", "true" if bool(bridge.get("shared_rgb_fallback_http", True)) else "false",
        "--camera-fps", str(int(bridge.get("fps", 30))),
        "--model-dir", str(model_dir),
        "--roi-config", str(runtime["roi_config_path"]),
        "--preprocess-backend", "auto",
        "--max-detections", str(int(runtime.get("max_detections", 20))),
        "--mask-max-points", str(int(algorithm["geometry"].get("contour_max_points", 96))),
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


def _app(config_path: str) -> int:
    command = [
        sys.executable,
        "-m",
        "production.carton_bundle_grasp.tasks.carton_bundle_grasp_vision.service",
        "--config",
        config_path,
    ]
    os.execv(command[0], command)
    return 0


def _collector(config: dict) -> int:
    task = config["carton_bundle_grasp"]
    collector = task["collector"]
    app = task["app"]
    os.environ["VISIONOPS_COLLECTOR_RUNTIME_SERVICE"] = "visionops-v3-carton-bundle-grasp-runtime.service"
    os.environ["VISIONOPS_COLLECTOR_CAMERA_DEPENDENT_SERVICES"] = "visionops-v3-carton-bundle-grasp-app.service"
    app_host = str(app["listen_host"])
    if app_host in {"0.0.0.0", "::"}:
        app_host = "127.0.0.1"
    app_url = "http://{}:{}".format(app_host, app["listen_port"])
    command = [
        sys.executable,
        "-m", "apps.collector_web.backend.main",
        "--host", str(collector["listen_host"]),
        "--port", str(collector["listen_port"]),
        "--runtime-url", str(task["runtime"]["url"]),
        "--gateway-url", app_url,
        "--business-app-url", app_url,
        "--production-inference-source", "app",
        "--snapshot-refresh-interval-ms", str(collector["snapshot_refresh_interval_ms"]),
        "--status-refresh-interval-ms", str(collector["status_refresh_interval_ms"]),
        "--device-id", str(collector["device_id"]),
        "--component", str(collector["component"]),
        "--models-root", str(collector["models_root"]),
    ]
    os.execv(command[0], command)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionOps M41 carton-bundle-grasp launcher")
    parser.add_argument("--config", help="默认 production/carton_bundle_grasp/config/line.yaml")
    parser.add_argument("command", choices=("runtime", "app", "collector", "show-config"))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = _config_path(args.config)
    config = load_config(path)
    if args.command == "runtime":
        return _runtime(config)
    if args.command == "app":
        return _app(path)
    if args.command == "collector":
        return _collector(config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
