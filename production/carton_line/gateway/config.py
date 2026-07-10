"""Configuration for the carton-line production services."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LINE_ROOT = Path(__file__).resolve().parents[1]


def _project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _line_path(*parts: str) -> str:
    return str(LINE_ROOT.joinpath(*parts))


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "kind": "production_line",
    "line_id": "carton_line",
    "device_id": "lb3576-production",
    "component": "carton_line_gateway",
    "service": {
        "listen_host": "127.0.0.1",
        "listen_port": 19090,
        "partition_app_port": 19120,
        "tube_app_port": 19110,
        "poll_interval_ms": 50,
        "request_timeout_ms": 5000,
    },
    "modbus": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5046,
        "unit_id": 1,
        "single_slave": True,
        "address_base": 0,
        "register_count": 200,
        "heartbeat_interval_ms": 500,
        "heartbeat_max": 1000,
    },
    "camera_bridge": {
        "base_url": "http://127.0.0.1:18182",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
        "depth_url": "http://127.0.0.1:18182/stream/depth.png",
    },
    "runtimes": {
        "partition": {
            "url": "http://127.0.0.1:28081",
            "model_dir": _project_path("models", "carton_partition_check", "current"),
            "device_id": "lb3576-partition",
            "component": "rknn_runtime_partition",
            "accepted_task_types": ["detection", "detect"],
            "accepted_model_ids": [],
            "accepted_model_names": [],
        },
        "tube": {
            "url": "http://127.0.0.1:28082",
            "model_dir": _project_path("models", "carton_tube_check", "current"),
            "device_id": "lb3576-tube",
            "component": "rknn_runtime_tube",
            "accepted_task_types": ["obb", "obb_detection"],
            "accepted_model_ids": [],
            "accepted_model_names": [],
        },
    },
    "collectors": {
        "partition": {
            "listen_host": "0.0.0.0",
            "listen_port": 18091,
            "device_id": "lb3576-partition",
            "component": "collector_partition",
            "models_root": _project_path("models", "carton_partition_check"),
            "snapshot_refresh_interval_ms": 200,
            "status_refresh_interval_ms": 2000,
        },
        "tube": {
            "listen_host": "0.0.0.0",
            "listen_port": 18092,
            "device_id": "lb3576-tube",
            "component": "collector_tube",
            "models_root": _project_path("models", "carton_tube_check"),
            "snapshot_refresh_interval_ms": 200,
            "status_refresh_interval_ms": 2000,
        },
    },
    "partition": {
        "template_path": _line_path(
            "tasks", "carton_partition_check", "assets", "partition_template.json"
        ),
        "algorithm": {
            "class_ids": [0],
            "class_names": ["cell", "paper_cell"],
            "min_confidence": 0.50,
            "nms_iou": 0.30,
            "expected_rows": 5,
            "expected_cols": 8,
            "expected_count": 40,
            "strict_count": True,
            "thresholds": {
                "max_mean_center_error_px": 22.0,
                "max_p95_center_error_px": 38.0,
                "max_center_error_px": 24.0,
                "max_edge_cell_error_px": 20.0,
                "max_row_angle_diff_max_deg": 1.0,
                "max_row_angle_std_diff_deg": 0.70,
                "max_grid_center_offset_px": 35.0,
                "max_row_angle_diff_deg": 5.0,
                "max_col_angle_diff_deg": 5.0,
                "max_affine_rotation_deg": 5.0,
                "max_affine_shear": 0.18,
            },
            "size_check": {
                "enabled": False,
                "min_ratio": 0.55,
                "max_ratio": 1.80,
                "max_bad_count": 6,
            },
        },
    },
    "tube": {
        "algorithm": {
            "stand_class_ids": [0],
            "lying_class_ids": [1],
            "stand_names": ["stand"],
            "lying_names": ["lying"],
            "min_confidence": 0.80,
            "min_stand_count": {"default": 1, "left": 1, "right": 1, "all": 1},
            "grid": {
                "rows": 5,
                "cols": 8,
                "slot_order": "col_major",
                "left_col_start": 0,
                "left_col_end": 3,
                "right_col_start": 4,
                "right_col_end": 7,
                "region_split_x": 0,
            },
            "depth": {
                "roi_radius_px": 12,
                "percentile": 50,
                "min_valid_pixels": 30,
                "min_depth_mm": 100,
                "max_depth_mm": 3000,
                "normal_depth_mm": 0,
                "baseline_mode": "row_median",
                "height_threshold_mm": 30,
            },
        },
    },
    "coordinates": {
        "output_frame": "robot",
        "register_order": "column",
        "always_ok": True,
        "partial_update_enabled": True,
        "partial_match_max_distance_px": 22.0,
        "partial_min_confidence": 0.10,
        "template_path": _line_path(
            "tasks", "carton_partition_check", "assets", "partition_template.json"
        ),
        "dual_arm_enabled": True,
        "left_columns": [0, 3],
        "right_columns": [4, 7],
        "single_affine": {"a00": 1.0, "a01": 0.0, "a10": 0.0, "a11": 1.0, "b0": 0.0, "b1": 0.0},
        "left_affine": {
            "a00": 0.02143055, "a01": -1.49495102,
            "a10": -1.47967273, "a11": -0.00292085,
            "b0": 946.29821487, "b1": 994.16131507,
        },
        "right_affine": {
            "a00": 0.00178128555, "a01": -1.50294475,
            "a10": -1.48032737, "a11": 0.000000792093091,
            "b0": 966.00553481, "b1": 994.4879298,
        },
    },
    "debug": {
        "save_every_trigger": True,
        "save_root": "/var/lib/visionops_v3/carton_line/latest",
    },
}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _valid_url(value: object, field: str) -> str:
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


def _project_relative_path(value: object) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_config(path: str | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        source = Path(path).expanduser().resolve()
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"无法读取产线配置 {source}: {error}") from error
        if not isinstance(document, Mapping):
            raise ValueError("产线配置顶层必须是对象")
        if document.get("kind") not in {None, "production_line"}:
            raise ValueError("产线配置 kind 必须为 production_line")
        config = _merge(config, document)

    service = config["service"]
    modbus = config["modbus"]
    for key in ("listen_port", "partition_app_port", "tube_app_port"):
        service[key] = _port(service[key], f"service.{key}")
    modbus["port"] = _port(modbus["port"], "modbus.port")

    used_ports = [service["listen_port"], service["partition_app_port"], service["tube_app_port"], modbus["port"]]
    for task in ("partition", "tube"):
        runtime = config["runtimes"][task]
        runtime["url"] = _valid_url(runtime["url"], f"runtimes.{task}.url")
        runtime_port = urlparse(runtime["url"]).port or (443 if runtime["url"].startswith("https:") else 80)
        used_ports.append(runtime_port)
        runtime["model_dir"] = _project_relative_path(runtime["model_dir"])
        runtime["accepted_task_types"] = [str(x).lower() for x in runtime.get("accepted_task_types", [])]
        runtime["accepted_model_ids"] = [str(x) for x in runtime.get("accepted_model_ids", []) if str(x)]
        runtime["accepted_model_names"] = [str(x) for x in runtime.get("accepted_model_names", []) if str(x)]

        collector = config["collectors"][task]
        collector["listen_port"] = _port(collector["listen_port"], f"collectors.{task}.listen_port")
        used_ports.append(collector["listen_port"])
        collector["models_root"] = _project_relative_path(collector["models_root"])
        for key in ("snapshot_refresh_interval_ms", "status_refresh_interval_ms"):
            collector[key] = int(collector[key])
            if collector[key] < 100:
                raise ValueError(f"collectors.{task}.{key} 不得小于 100")

    if len(used_ports) != len(set(used_ports)):
        raise ValueError("Runtime、Collector、Gateway、业务兼容接口和 Modbus 端口必须互不相同")

    for key in ("poll_interval_ms", "request_timeout_ms"):
        service[key] = int(service[key])
        if service[key] <= 0:
            raise ValueError(f"service.{key} 必须大于 0")
    for key in ("heartbeat_interval_ms", "heartbeat_max", "register_count"):
        modbus[key] = int(modbus[key])
        if modbus[key] <= 0:
            raise ValueError(f"modbus.{key} 必须大于 0")
    modbus["unit_id"] = int(modbus.get("unit_id", 1))
    if not 0 <= modbus["unit_id"] <= 255:
        raise ValueError("modbus.unit_id 必须位于 0..255")
    modbus["single_slave"] = bool(modbus.get("single_slave", True))
    modbus["address_base"] = int(modbus.get("address_base", 0))
    if modbus["address_base"] < 0 or modbus["register_count"] < 200:
        raise ValueError("modbus.address_base 必须非负且 register_count 至少为 200")

    bridge = config["camera_bridge"]
    bridge["base_url"] = _valid_url(bridge["base_url"], "camera_bridge.base_url")
    bridge["depth_url"] = _valid_url(bridge["depth_url"], "camera_bridge.depth_url")
    config["partition"]["template_path"] = _project_relative_path(config["partition"]["template_path"])
    config["coordinates"]["template_path"] = _project_relative_path(config["coordinates"]["template_path"])
    config["debug"]["save_root"] = str(Path(config["debug"]["save_root"]).expanduser())
    return config
