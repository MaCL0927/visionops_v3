#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loader for the independent plastic-bag grasp production task."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

import yaml

from edge.camera_bridge.camera_selection import active_camera_spec, public_mjpeg_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "production/plastic_bag_grasp/config/line.yaml"


def _project_path(*parts: str) -> str:
    return str((PROJECT_ROOT.joinpath(*parts)).resolve())


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "task_id": "plastic_bag_grasp",
    "device_id": "lb3576-plastic-bag-grasp",
    "camera_bridge": {
        "camera_model": "orbbec336l",
        "base_url": "http://127.0.0.1:18182",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
        "mjpeg_path": "/stream.mjpeg",
        "sample_deproject_path": "/api/coordinate/sample_deproject",
        "fps": 30,
        "shared_rgb_enabled": True,
        "shared_rgb_name": "/visionops_orbbec336l_rgb",
        "shared_rgb_fallback_http": True,
        "shared_depth_enabled": True,
        "shared_depth_name": "/visionops_orbbec336l_depth",
        "shared_depth_fallback_http": True,
        "stale_ms": 5000,
    },
    "runtime_recovery": {
        "stale_frame_timeout_ms": 3000,
        "failure_threshold": 3,
        "initial_backoff_ms": 200,
        "max_backoff_ms": 2000,
    },
    "runtime": {
        "url": "http://127.0.0.1:28088",
        # Current model seen in the M40 validation screenshot.  Override with
        # VISIONOPS_PLASTIC_BAG_GRASP_MODEL_DIR or edit line.yaml when replaced.
        "model_dir": _project_path("models", "rk3576-252_plastic_bag_grasp_det_20260819_094048"),
        "roi_config_path": _project_path("data", "runtime", "roi_plastic_bag_grasp.json"),
        "device_id": "lb3576-plastic-bag-grasp-runtime",
        "component": "rknn_runtime_plastic_bag_grasp",
        "accepted_task_types": ["detection", "detect"],
        "accepted_model_ids": [],
        "accepted_model_names": [],
        "max_detections": 20,
    },
    "app": {
        "listen_host": "127.0.0.1",
        "listen_port": 19214,
        "request_timeout_ms": 5000,
        "inference_settings_path": _project_path(
            "configs", "runtime", "generated", "plastic_bag_grasp_inference_settings.json"
        ),
        # The Runtime shown by the user is already configured for 30 FPS and
        # reaches ~22.6 FPS.  The App must not impose a legacy 5 Hz throttle.
        "default_production_inference_fps": 30.0,
    },
    "collector": {
        "listen_host": "0.0.0.0",
        "listen_port": 18097,
        "device_id": "lb3576-plastic-bag-grasp",
        "component": "collector_plastic_bag_grasp",
        "models_root": _project_path("models"),
        "snapshot_refresh_interval_ms": 200,
        "status_refresh_interval_ms": 2000,
        "production_inference_source": "app",
    },
    "websocket": {
        "listen_host": "0.0.0.0",
        "listen_port": 9001,
        "path": "/vision",
        "token": "",
        "auto_start": True,
        "status_interval_s": 2.0,
        "read_timeout_s": 30.0,
        "max_clients": 4,
        "max_payload_bytes": 1048576,
        "trigger_queue_size": 32,
        "trigger_task_ids": ["plastic_bag_grasp", "plastic_bag_pick", "1", 1],
    },
    "pipeline": {
        # Validated M32.8 architecture: Runtime inference and CPU postprocess
        # overlap through a capacity-1 latest-only queue.  Trigger packets are
        # protected from continuous-frame replacement.
        "enabled": True,
        "result_queue_size": 1,
        "max_result_age_ms": 500,
    },
    "runtime_ipc": {
        # Validated M32.8.1 local Runtime path: TCP_NODELAY + one sendall.  Raw
        # response bytes are decoded by the postprocess thread, not the producer.
        "raw_http_enabled": True,
        "raw_http_fallback_urllib": True,
        "max_response_bytes": 32 * 1024 * 1024,
    },
    "video": {
        "type": "mjpeg",
        "public_url": "http://192.168.20.20:18182/stream.mjpeg",
        "sync": "soft",
    },
    "algorithm": {
        "image": {"width": 640, "height": 480, "require_fixed_size": False},
        "classes": {
            "target_ids": [0],
            "target_names": ["plastic_bag", "bag", "plastic_bag_package", "package"],
            "min_confidence": 0.50,
        },
        "selection": {
            # Production scene contains one package.  If duplicate detections
            # survive Runtime NMS, keep only the highest-confidence target.
            "max_targets": 1,
            "mode": "confidence",
        },
        "depth": {
            # Pixel centre is always authoritative for robot-side 2-D calibration.
            # Camera XYZ is opportunistically added from D2C depth.  Invalid depth
            # never suppresses a valid RGB detection; position_camera becomes zero.
            "enabled": True,
            "roi_radius_px": 6,
            "percentile": 50.0,
            "min_valid_pixels": 5,
            "min_depth_mm": 100,
            "max_depth_mm": 5000,
            "max_age_ms": 1500,
        },
    },
    "debug": {
        "save_every_trigger": False,
        "save_root": "/tmp/visionops_v3/plastic_bag_grasp/latest",
    },
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def _path(value: object) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _url(value: object, field: str) -> str:
    text = str(value or "").rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} 必须是 HTTP/HTTPS URL")
    return text


def _port(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须是端口整数")
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError(f"{field} 必须位于 1..65535")
    return number


def _apply_active_camera(config: Dict[str, Any]) -> None:
    spec = active_camera_spec()
    bridge = config["camera_bridge"]
    bridge["camera_model"] = spec["camera_model"]
    bridge["base_url"] = spec["base_url"]
    for key in ("snapshot_path", "health_path", "mjpeg_path"):
        if key in spec:
            bridge[key] = spec[key]
    bridge["service"] = spec["service"]
    config["video"]["public_url"] = public_mjpeg_url(config["video"].get("public_url"), spec)
    config["active_camera"] = {
        "camera_model": spec["camera_model"],
        "display_name": spec["display_name"],
        "base_url": spec["base_url"],
        "service": spec["service"],
        "selection_path": spec["selection_path"],
    }


def _validate(config: Dict[str, Any]) -> None:
    runtime = config["runtime"]
    runtime["url"] = _url(runtime["url"], "runtime.url")
    runtime["model_dir"] = _path(runtime["model_dir"])
    runtime["roi_config_path"] = _path(runtime["roi_config_path"])
    runtime["accepted_task_types"] = [
        str(item).strip().lower() for item in runtime.get("accepted_task_types", []) if str(item).strip()
    ]
    runtime["accepted_model_ids"] = [str(item) for item in runtime.get("accepted_model_ids", []) if str(item)]
    runtime["accepted_model_names"] = [str(item) for item in runtime.get("accepted_model_names", []) if str(item)]
    if not runtime["accepted_task_types"]:
        raise ValueError("runtime.accepted_task_types 不能为空")
    runtime["max_detections"] = max(1, int(runtime.get("max_detections", 20)))

    app = config["app"]
    app["listen_port"] = _port(app["listen_port"], "app.listen_port")
    app["request_timeout_ms"] = max(100, int(app.get("request_timeout_ms", 5000)))
    app["inference_settings_path"] = _path(app["inference_settings_path"])
    default_fps = float(app.get("default_production_inference_fps", 30.0))
    if not 0.1 <= default_fps <= 30.0:
        raise ValueError("app.default_production_inference_fps 必须位于 0.1..30")
    app["default_production_inference_fps"] = default_fps

    collector = config["collector"]
    collector["listen_port"] = _port(collector["listen_port"], "collector.listen_port")
    collector["models_root"] = _path(collector["models_root"])
    collector["snapshot_refresh_interval_ms"] = max(100, int(collector.get("snapshot_refresh_interval_ms", 200)))
    collector["status_refresh_interval_ms"] = max(100, int(collector.get("status_refresh_interval_ms", 2000)))
    if str(collector.get("production_inference_source", "app")) != "app":
        raise ValueError("collector.production_inference_source 必须为 app")

    websocket = config["websocket"]
    websocket["listen_port"] = _port(websocket["listen_port"], "websocket.listen_port")
    websocket["path"] = str(websocket.get("path") or "/vision")
    if not websocket["path"].startswith("/"):
        websocket["path"] = "/" + websocket["path"]
    websocket["trigger_queue_size"] = max(1, int(websocket.get("trigger_queue_size", 32)))
    websocket["max_clients"] = max(1, int(websocket.get("max_clients", 4)))
    websocket["read_timeout_s"] = max(1.0, float(websocket.get("read_timeout_s", 30.0)))
    websocket["status_interval_s"] = max(0.5, float(websocket.get("status_interval_s", 2.0)))
    websocket["max_payload_bytes"] = max(1024, int(websocket.get("max_payload_bytes", 1048576)))

    pipeline = config["pipeline"]
    pipeline["enabled"] = bool(pipeline.get("enabled", True))
    pipeline["result_queue_size"] = max(1, int(pipeline.get("result_queue_size", 1)))
    pipeline["max_result_age_ms"] = max(1, int(pipeline.get("max_result_age_ms", 500)))

    ipc = config["runtime_ipc"]
    ipc["raw_http_enabled"] = bool(ipc.get("raw_http_enabled", True))
    ipc["raw_http_fallback_urllib"] = bool(ipc.get("raw_http_fallback_urllib", True))
    ipc["max_response_bytes"] = max(1024, int(ipc.get("max_response_bytes", 32 * 1024 * 1024)))

    bridge = config["camera_bridge"]
    bridge["sample_deproject_path"] = str(bridge.get("sample_deproject_path") or "/api/coordinate/sample_deproject")
    if not bridge["sample_deproject_path"].startswith("/"):
        bridge["sample_deproject_path"] = "/" + bridge["sample_deproject_path"]
    bridge["shared_depth_enabled"] = bool(bridge.get("shared_depth_enabled", True))
    bridge["shared_depth_fallback_http"] = bool(bridge.get("shared_depth_fallback_http", True))

    config["video"]["public_url"] = _url(config["video"]["public_url"], "video.public_url")

    image = config["algorithm"]["image"]
    image["width"] = max(1, int(image.get("width", 640)))
    image["height"] = max(1, int(image.get("height", 480)))

    classes = config["algorithm"]["classes"]
    classes["target_ids"] = [int(item) for item in classes.get("target_ids", [])]
    classes["target_names"] = [str(item).strip().lower() for item in classes.get("target_names", []) if str(item).strip()]
    if not classes["target_ids"] and not classes["target_names"]:
        raise ValueError("algorithm.classes 至少配置一个 target_id 或 target_name")
    classes["min_confidence"] = float(classes.get("min_confidence", 0.5))
    if not 0.0 <= classes["min_confidence"] <= 1.0:
        raise ValueError("algorithm.classes.min_confidence 必须位于 0..1")

    selection = config["algorithm"]["selection"]
    selection["max_targets"] = max(1, int(selection.get("max_targets", 1)))
    selection["mode"] = str(selection.get("mode", "confidence")).strip().lower()
    if selection["mode"] not in {"confidence", "largest_area"}:
        raise ValueError("algorithm.selection.mode 必须为 confidence/largest_area")

    depth = config["algorithm"]["depth"]
    depth["enabled"] = bool(depth.get("enabled", True))
    depth["roi_radius_px"] = max(0, int(depth.get("roi_radius_px", 6)))
    depth["percentile"] = min(100.0, max(0.0, float(depth.get("percentile", 50.0))))
    depth["min_valid_pixels"] = max(1, int(depth.get("min_valid_pixels", 5)))
    depth["min_depth_mm"] = max(0, int(depth.get("min_depth_mm", 100)))
    depth["max_depth_mm"] = max(depth["min_depth_mm"] + 1, int(depth.get("max_depth_mm", 5000)))
    depth["max_age_ms"] = max(1, int(depth.get("max_age_ms", 1500)))

    ports = {
        urlparse(runtime["url"]).port or 80,
        app["listen_port"],
        collector["listen_port"],
        websocket["listen_port"],
    }
    if len(ports) != 4:
        raise ValueError("Runtime/App/Collector/WebSocket 端口不得重复")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    target = Path(path).expanduser()
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loaded = {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError("plastic_bag_grasp 配置顶层必须是对象")
    config = _merge(DEFAULT_CONFIG, loaded)
    _apply_active_camera(config)
    _validate(config)
    config["config_path"] = str(target.resolve())
    return config
