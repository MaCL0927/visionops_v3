#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loader for the independent detergent-grasp production task."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

import yaml

from edge.camera_bridge.camera_selection import active_camera_spec, public_mjpeg_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "production/detergent_grasp/config/line.yaml"


def _project_path(*parts: str) -> str:
    return str((PROJECT_ROOT.joinpath(*parts)).resolve())


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "task_id": "detergent_grasp",
    "device_id": "lb3576-detergent-grasp",
    "camera_bridge": {
        "camera_model": "hp60c",
        "base_url": "http://127.0.0.1:18181",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
        "mjpeg_path": "/stream.mjpeg",
        "fps": 30,
        "shared_rgb_enabled": True,
        "shared_rgb_name": "/visionops_orbbec336l_rgb",
        "shared_rgb_fallback_http": True,
        "stale_ms": 5000,
    },
    "runtime_recovery": {
        "stale_frame_timeout_ms": 3000,
        "failure_threshold": 3,
        "initial_backoff_ms": 200,
        "max_backoff_ms": 2000,
    },
    "runtime": {
        "url": "http://127.0.0.1:28087",
        "model_dir": _project_path("models", "detergent_grasp", "current"),
        "roi_config_path": _project_path("data", "runtime", "roi_detergent_grasp.json"),
        "device_id": "lb3576-detergent-grasp-runtime",
        "component": "rknn_runtime_detergent_grasp",
        "accepted_task_types": ["obb"],
        "max_detections": 100,
    },
    "app": {
        "listen_host": "127.0.0.1",
        "listen_port": 19212,
        "request_timeout_ms": 5000,
        "inference_settings_path": "/opt/visionops_v3/config/detergent_grasp_inference_settings.json",
        "default_production_inference_fps": 15.0,
    },
    "collector": {
        "listen_host": "0.0.0.0",
        "listen_port": 18096,
        "device_id": "lb3576-detergent-grasp",
        "component": "collector_detergent_grasp",
        "models_root": _project_path("models", "detergent_grasp"),
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
        "trigger_task_ids": ["detergent_grasp", "detergent_pick", "1", 1],
    },
    "video": {
        "type": "mjpeg",
        "public_url": "http://192.168.20.20:18181/stream.mjpeg",
        "sync": "soft",
    },
    "algorithm": {
        "image": {"width": 640, "height": 480, "require_fixed_size": True},
        "require_obb": True,
        "classes": {
            # Current training convention: 0=big, 1=head, 2=box, 3=small.
            # Names take precedence so class-ID reordering is still safe.
            "big_bottle_ids": [0],
            "big_bottle_names": ["big", "large", "big_bottle", "large_bottle"],
            "small_bottle_ids": [3],
            "small_bottle_names": ["small", "small_bottle"],
            "grasp_point_ids": [1],
            "grasp_point_names": ["head", "grasp", "grasp_point", "pick_point"],
            "box_ids": [2],
            "box_names": ["box", "carton", "destination_box"],
            "big_bottle_min_confidence": 0.50,
            "small_bottle_min_confidence": 0.50,
            "grasp_point_min_confidence": 0.50,
            "box_min_confidence": 0.50,
        },
        "association": {
            "bottle_polygon_expand_ratio": 1.18,
            "max_center_distance_ratio": 0.65,
            "require_grasp_point": True,
        },
        "selection": {
            "max_bottles": 8,
            "max_boxes": 1,
            "output_order": "row_major",
        },
        "output": {
            # Bottle angle_deg is resolved to a directed handle orientation in
            # [-180, 180) without changing the robot-facing field name.
            "include_obb_points": True,
            "include_class_name": True,
        },
    },
    "debug": {
        "save_every_trigger": False,
        "save_root": "/tmp/visionops_v3/detergent_grasp/latest",
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
        raise ValueError("{} 必须是 HTTP/HTTPS URL".format(field))
    return text


def _port(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} 必须是端口整数".format(field))
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError("{} 必须位于 1..65535".format(field))
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
    if not runtime["accepted_task_types"]:
        raise ValueError("runtime.accepted_task_types 不能为空")
    runtime["max_detections"] = max(1, int(runtime.get("max_detections", 100)))

    app = config["app"]
    app["listen_port"] = _port(app["listen_port"], "app.listen_port")
    app["request_timeout_ms"] = max(100, int(app.get("request_timeout_ms", 5000)))
    app["inference_settings_path"] = _path(app["inference_settings_path"])
    default_fps = float(app.get("default_production_inference_fps", 15.0))
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

    video = config["video"]
    video["public_url"] = _url(video["public_url"], "video.public_url")

    image = config["algorithm"]["image"]
    image["width"] = max(1, int(image.get("width", 640)))
    image["height"] = max(1, int(image.get("height", 480)))
    classes = config["algorithm"]["classes"]
    for key in ("big_bottle_ids", "small_bottle_ids", "grasp_point_ids", "box_ids"):
        classes[key] = [int(item) for item in classes.get(key, [])]
    for key in ("big_bottle_names", "small_bottle_names", "grasp_point_names", "box_names"):
        classes[key] = [str(item).strip().lower() for item in classes.get(key, []) if str(item).strip()]
    for semantic in ("big_bottle", "small_bottle", "grasp_point", "box"):
        if not classes.get(semantic + "_ids") and not classes.get(semantic + "_names"):
            raise ValueError("algorithm.classes.{} 至少配置一个 ID 或名称".format(semantic))
        threshold_key = semantic + "_min_confidence"
        threshold = float(classes.get(threshold_key, 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("algorithm.classes.{} 必须位于 0..1".format(threshold_key))
        classes[threshold_key] = threshold

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
        raise ValueError("detergent_grasp 配置顶层必须是对象")
    config = _merge(DEFAULT_CONFIG, loaded)
    _apply_active_camera(config)
    _validate(config)
    config["config_path"] = str(target.resolve())
    return config
