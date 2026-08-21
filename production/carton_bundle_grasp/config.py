#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration for M41 carton-bundle top-plane grasp task."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Union
from urllib.parse import urlparse

import yaml

from edge.camera_bridge.camera_selection import active_camera_spec, public_mjpeg_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "production/carton_bundle_grasp/config/line.yaml"


def _project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts).resolve())


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "camera_bridge": {
        "camera_model": "orbbec336l",
        "base_url": "http://127.0.0.1:18182",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
        "mjpeg_path": "/stream.mjpeg",
        "depth_path": "/stream/depth.png",
        "deproject_path": "/api/coordinate/deproject",
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
    "carton_bundle_grasp": {
        "device_id": "lb3576-carton-bundle-grasp",
        "component": "carton_bundle_grasp_app",
        "runtime": {
            "url": "http://127.0.0.1:28089",
            # M41 code is ready before the production RKNN package is named.
            # Put model.rknn + model.yaml here or override with the env variable.
            "model_dir": _project_path("models", "carton_bundle_grasp", "current"),
            "roi_config_path": _project_path("data", "runtime", "roi_carton_bundle_grasp.json"),
            "device_id": "lb3576-carton-bundle-grasp-runtime",
            "component": "rknn_runtime_carton_bundle_grasp",
            "accepted_task_types": ["segmentation", "segment"],
            "max_detections": 20,
        },
        "app": {
            "listen_host": "127.0.0.1",
            "listen_port": 19215,
            "request_timeout_ms": 5000,
            "inference_settings_path": _project_path(
                "configs", "runtime", "generated", "carton_bundle_grasp_inference_settings.json"
            ),
            # App target only; every completed result is pushed. There is no
            # separate fixed 5-Hz WebSocket throttle.
            "default_production_inference_fps": 30.0,
        },
        "pipeline": {
            "enabled": True,
            "result_queue_size": 1,
            "max_result_age_ms": 500,
        },
        "ipc": {
            "raw_http_enabled": True,
            "raw_http_fallback_urllib": True,
            "max_response_bytes": 32 * 1024 * 1024,
        },
        "collector": {
            "listen_host": "0.0.0.0",
            "listen_port": 18098,
            "device_id": "lb3576-carton-bundle-grasp",
            "component": "collector_carton_bundle_grasp",
            "models_root": _project_path("models", "carton_bundle_grasp"),
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
        },
        "video": {
            "type": "mjpeg",
            "public_url": "http://192.168.20.20:18182/stream.mjpeg",
            "sync": "soft",
        },
        "algorithm": {
            "image": {"width": 1280, "height": 720, "require_fixed_size": False},
            "classes": {
                "target_ids": [0],
                "target_names": ["carton_bundle_top", "bundle_top", "carton_bundle", "box"],
                "min_confidence": 0.50,
            },
            "selection": {
                # M41 simple scene: the highest-confidence complete top face.
                "max_targets": 1,
                "mode": "confidence",
            },
            "geometry": {
                "require_proto_mask": True,
                "min_mask_area_px": 2500,
                "epsilon_min": 0.006,
                "epsilon_max": 0.12,
                "epsilon_steps": 28,
                "min_quad_area_ratio": 0.62,
                "max_quad_area_ratio": 1.38,
                "contour_max_points": 96,
            },
            "bundle_prior": {
                "length_mm": 715.0,
                "width_mm": 525.0,
                # First production gate is intentionally tolerant of mask edge
                # bias; fixed-size regularisation still outputs exact prior size.
                "length_tolerance_mm": 80.0,
                "width_tolerance_mm": 70.0,
                "regularize_fixed_size": True,
            },
            "top_plane": {
                "sample_count": 96,
                "erode_px": 18,
                "ransac_threshold_mm": 5.0,
                "ransac_trials": 96,
                "min_valid_samples": 36,
                "min_inlier_ratio": 0.70,
                "max_rms_mm": 6.0,
            },
            "depth": {
                "enabled": True,
                # Plane samples are intentionally low-radius and distributed.
                "roi_radius_px": 2,
                "percentile": 50.0,
                "min_valid_pixels": 1,
                "min_depth_mm": 100,
                "max_depth_mm": 5000,
                "max_age_ms": 1500,
            },
            "robot_state": {
                # Camera XYZ reconstruction does not depend on robot SDK pose.
                # Future waist_z_mm can be attached here after SDK API details
                # are supplied; do not compensate camera XYZ in this module.
                "waist_z_required_for_geometry": False,
                "sdk_read_enabled": False,
            },
        },
        "debug": {
            "save_every_trigger": False,
            "save_root": "/tmp/visionops_v3/carton_bundle_grasp/latest",
        },
    },
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


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
    task = config["carton_bundle_grasp"]
    task["video"]["public_url"] = public_mjpeg_url(task["video"].get("public_url"), spec)
    config["active_camera"] = {
        "camera_model": spec["camera_model"],
        "display_name": spec["display_name"],
        "base_url": spec["base_url"],
        "service": spec["service"],
        "selection_path": spec["selection_path"],
    }


def _validate(config: Dict[str, Any]) -> None:
    task = config["carton_bundle_grasp"]
    runtime = task["runtime"]
    runtime["url"] = _url(runtime["url"], "runtime.url")
    runtime["model_dir"] = _path(runtime["model_dir"])
    runtime["roi_config_path"] = _path(runtime["roi_config_path"])
    runtime["accepted_task_types"] = [
        str(item).strip().lower() for item in runtime.get("accepted_task_types", []) if str(item).strip()
    ]
    runtime["max_detections"] = max(1, int(runtime.get("max_detections", 20)))

    app = task["app"]
    app["listen_port"] = _port(app["listen_port"], "app.listen_port")
    app["request_timeout_ms"] = max(100, int(app.get("request_timeout_ms", 5000)))
    app["inference_settings_path"] = _path(app["inference_settings_path"])
    app["default_production_inference_fps"] = float(app.get("default_production_inference_fps", 30.0))
    if not 0.1 <= app["default_production_inference_fps"] <= 30.0:
        raise ValueError("app.default_production_inference_fps 必须位于 0.1..30")

    collector = task["collector"]
    collector["listen_port"] = _port(collector["listen_port"], "collector.listen_port")
    collector["models_root"] = _path(collector["models_root"])

    websocket = task["websocket"]
    websocket["listen_port"] = _port(websocket["listen_port"], "websocket.listen_port")
    websocket["path"] = str(websocket.get("path") or "/vision")
    if not websocket["path"].startswith("/"):
        websocket["path"] = "/" + websocket["path"]
    websocket["trigger_queue_size"] = max(1, int(websocket.get("trigger_queue_size", 32)))

    pipeline = task["pipeline"]
    pipeline["enabled"] = bool(pipeline.get("enabled", True))
    pipeline["result_queue_size"] = max(1, int(pipeline.get("result_queue_size", 1)))
    pipeline["max_result_age_ms"] = max(1, int(pipeline.get("max_result_age_ms", 500)))

    algorithm = task["algorithm"]
    prior = algorithm["bundle_prior"]
    prior["length_mm"] = float(prior.get("length_mm", 715.0))
    prior["width_mm"] = float(prior.get("width_mm", 525.0))
    if prior["length_mm"] <= prior["width_mm"] or prior["width_mm"] <= 0:
        raise ValueError("bundle_prior 必须满足 length_mm > width_mm > 0")
    classes = algorithm["classes"]
    classes["target_ids"] = [int(item) for item in classes.get("target_ids", [])]
    classes["target_names"] = [str(item).strip().lower() for item in classes.get("target_names", []) if str(item).strip()]

    bridge = config["camera_bridge"]
    for key, default in (
        ("deproject_path", "/api/coordinate/deproject"),
        ("sample_deproject_path", "/api/coordinate/sample_deproject"),
    ):
        bridge[key] = str(bridge.get(key) or default)
        if not bridge[key].startswith("/"):
            bridge[key] = "/" + bridge[key]

    task["video"]["public_url"] = _url(task["video"]["public_url"], "video.public_url")
    ports = {
        urlparse(runtime["url"]).port or 80,
        app["listen_port"],
        collector["listen_port"],
        websocket["listen_port"],
    }
    if len(ports) != 4:
        raise ValueError("Runtime/App/Collector/WebSocket 端口不得重复")


def load_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    target = Path(path).expanduser()
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loaded = {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError("carton_bundle_grasp 配置顶层必须是对象")
    config = _merge(DEFAULT_CONFIG, loaded)
    _apply_active_camera(config)
    _validate(config)
    config["config_path"] = str(target.resolve())
    return config
