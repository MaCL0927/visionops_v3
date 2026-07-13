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
    "runtime_recovery": {
        "stale_frame_timeout_ms": 3000,
        "failure_threshold": 3,
        "initial_backoff_ms": 200,
        "max_backoff_ms": 2000,
    },
    "camera_bridge": {
        "base_url": "http://127.0.0.1:18182",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
        "depth_url": "http://127.0.0.1:18182/stream/depth.png",
        "depth_meta_url": "http://127.0.0.1:18182/stream/depth_meta",
        "deproject_url": "http://127.0.0.1:18182/api/coordinate/deproject",
    },
    "runtimes": {
        "partition": {
            "url": "http://127.0.0.1:28081",
            "model_dir": _project_path("models", "carton_partition_check", "current"),
            "roi_config_path": _project_path("data", "runtime", "roi_partition.json"),
            "device_id": "lb3576-partition",
            "component": "rknn_runtime_partition",
            "accepted_task_types": ["detection", "detect"],
            "accepted_model_ids": [],
            "accepted_model_names": [],
        },
        "tube": {
            "url": "http://127.0.0.1:28082",
            "model_dir": _project_path("models", "carton_tube_check", "current"),
            "roi_config_path": _project_path("data", "runtime", "roi_tube.json"),
            "device_id": "lb3576-tube",
            "component": "rknn_runtime_tube",
            "accepted_task_types": ["obb", "obb_detection"],
            "accepted_model_ids": [],
            "accepted_model_names": [],
        },
        "pick": {
            "url": "http://127.0.0.1:28083",
            "model_dir": _project_path("models", "tube_pick_vision", "current"),
            "roi_config_path": _project_path("data", "runtime", "roi_pick.json"),
            "device_id": "lb3576-tube-pick",
            "component": "rknn_runtime_tube_pick",
            "accepted_task_types": ["detection", "detect"],
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
        "pick": {
            "listen_host": "0.0.0.0",
            "listen_port": 18093,
            "device_id": "lb3576-tube-pick",
            "component": "collector_tube_pick",
            "models_root": _project_path("models", "tube_pick_vision"),
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
    "pick": {
        "websocket": {
            "listen_host": "0.0.0.0",
            "listen_port": 9001,
            "path": "/vision",
            "token": "",
            "auto_start": True,
            "detection_hz": 10.0,
            "status_interval_s": 2.0,
            "read_timeout_s": 30.0,
            "max_clients": 4,
            "max_payload_bytes": 1048576,
            "trigger_queue_size": 32,
        },
        "http": {"listen_host": "127.0.0.1", "listen_port": 19130},
        "video": {
            "type": "mjpeg",
            "public_url": "http://192.168.2.211:18182/stream.mjpeg",
            "sync": "soft",
        },
        "algorithm": {
            "image": {
                "width": 640,
                "height": 480,
                "require_fixed_size": True,
            },
            "classes": {
                "product_ids": [0],
                "separator_ids": [1],
                "product_names": ["tube_product", "product", "tube"],
                "separator_names": ["large_separator", "separator", "partition"],
                "product_min_confidence": 0.50,
                "separator_min_confidence": 0.50,
                "output_order": "row_major",
            },
            "depth": {
                "roi_radius_px": 4,
                "percentile": 50,
                "min_valid_pixels": 3,
                "min_depth_mm": 100,
                "max_depth_mm": 5000,
                "max_age_ms": 1500,
            },
        },
        "debug": {
            "save_every_trigger": True,
            "save_root": "/tmp/visionops_v3/carton_line/tube_pick_vision/latest",
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
        "save_root": "/tmp/visionops_v3/carton_line/latest",
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

    pick_ws = config["pick"]["websocket"]
    pick_http = config["pick"]["http"]
    pick_ws["listen_port"] = _port(pick_ws["listen_port"], "pick.websocket.listen_port")
    pick_http["listen_port"] = _port(pick_http["listen_port"], "pick.http.listen_port")

    used_ports = [
        service["listen_port"], service["partition_app_port"], service["tube_app_port"],
        modbus["port"], pick_ws["listen_port"], pick_http["listen_port"],
    ]
    for task in ("partition", "tube", "pick"):
        runtime = config["runtimes"][task]
        runtime["url"] = _valid_url(runtime["url"], f"runtimes.{task}.url")
        runtime_port = urlparse(runtime["url"]).port or (443 if runtime["url"].startswith("https:") else 80)
        used_ports.append(runtime_port)
        runtime["model_dir"] = _project_relative_path(runtime["model_dir"])
        runtime["roi_config_path"] = _project_relative_path(
            runtime.get("roi_config_path") or f"data/runtime/roi_{task}.json"
        )
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

    recovery = config["runtime_recovery"]
    for key in ("stale_frame_timeout_ms", "failure_threshold", "initial_backoff_ms", "max_backoff_ms"):
        recovery[key] = int(recovery[key])
        if recovery[key] <= 0:
            raise ValueError(f"runtime_recovery.{key} 必须大于 0")
    if recovery["max_backoff_ms"] < recovery["initial_backoff_ms"]:
        raise ValueError("runtime_recovery.max_backoff_ms 不得小于 initial_backoff_ms")

    bridge = config["camera_bridge"]
    bridge["base_url"] = _valid_url(bridge["base_url"], "camera_bridge.base_url")
    bridge["depth_url"] = _valid_url(bridge["depth_url"], "camera_bridge.depth_url")
    bridge["depth_meta_url"] = _valid_url(bridge["depth_meta_url"], "camera_bridge.depth_meta_url")
    bridge["deproject_url"] = _valid_url(bridge["deproject_url"], "camera_bridge.deproject_url")
    config["partition"]["template_path"] = _project_relative_path(config["partition"]["template_path"])
    config["coordinates"]["template_path"] = _project_relative_path(config["coordinates"]["template_path"])
    config["debug"]["save_root"] = str(Path(config["debug"]["save_root"]).expanduser())

    pick_ws["listen_host"] = str(pick_ws.get("listen_host") or "0.0.0.0")
    pick_ws["path"] = str(pick_ws.get("path") or "/vision")
    if not pick_ws["path"].startswith("/"):
        pick_ws["path"] = "/" + pick_ws["path"]
    pick_ws["token"] = str(pick_ws.get("token") or "")
    pick_ws["auto_start"] = bool(pick_ws.get("auto_start", True))
    for key in ("max_clients", "max_payload_bytes", "trigger_queue_size"):
        pick_ws[key] = int(pick_ws[key])
        if pick_ws[key] <= 0:
            raise ValueError(f"pick.websocket.{key} 必须大于 0")
    for key in ("detection_hz", "status_interval_s", "read_timeout_s"):
        pick_ws[key] = float(pick_ws[key])
        if pick_ws[key] <= 0:
            raise ValueError(f"pick.websocket.{key} 必须大于 0")
    if pick_ws["detection_hz"] > 30.0:
        raise ValueError("pick.websocket.detection_hz 不得大于 30")
    pick_http["listen_host"] = str(pick_http.get("listen_host") or "127.0.0.1")

    video = config["pick"]["video"]
    video["type"] = str(video.get("type") or "mjpeg").lower()
    if video["type"] != "mjpeg":
        raise ValueError("pick.video.type 当前只支持 mjpeg")
    video["public_url"] = _valid_url(video["public_url"], "pick.video.public_url")
    video["sync"] = "soft"

    pick_algorithm = config["pick"]["algorithm"]
    image_config = pick_algorithm["image"]
    for key in ("width", "height"):
        image_config[key] = int(image_config[key])
        if image_config[key] <= 0:
            raise ValueError(f"pick.algorithm.image.{key} 必须大于 0")
    image_config["require_fixed_size"] = bool(image_config.get("require_fixed_size", True))
    class_config = pick_algorithm["classes"]
    product_ids = {int(value) for value in class_config.get("product_ids", [])}
    separator_ids = {int(value) for value in class_config.get("separator_ids", [])}
    if not product_ids or not separator_ids:
        raise ValueError("pick.algorithm.classes 的 product_ids 和 separator_ids 不能为空")
    if product_ids & separator_ids:
        raise ValueError("pick.algorithm.classes 的 product_ids 和 separator_ids 不得重叠")
    class_config["product_ids"] = sorted(product_ids)
    class_config["separator_ids"] = sorted(separator_ids)
    for key in ("product_min_confidence", "separator_min_confidence"):
        class_config[key] = float(class_config[key])
        if not 0.0 <= class_config[key] <= 1.0:
            raise ValueError(f"pick.algorithm.classes.{key} 必须位于 0..1")
    class_config["output_order"] = str(class_config.get("output_order", "row_major")).lower()
    if class_config["output_order"] not in {"row_major", "column_major", "confidence"}:
        raise ValueError("pick.algorithm.classes.output_order 必须是 row_major/column_major/confidence")

    depth_config = pick_algorithm["depth"]
    for key in ("roi_radius_px", "min_valid_pixels", "min_depth_mm", "max_depth_mm", "max_age_ms"):
        depth_config[key] = int(depth_config[key])
    depth_config["percentile"] = float(depth_config["percentile"])
    if depth_config["roi_radius_px"] < 0 or depth_config["min_valid_pixels"] <= 0:
        raise ValueError("pick.algorithm.depth 的 roi_radius_px/min_valid_pixels 非法")
    if depth_config["min_depth_mm"] < 0 or depth_config["max_depth_mm"] <= depth_config["min_depth_mm"]:
        raise ValueError("pick.algorithm.depth 的深度范围非法")
    if not 0.0 <= depth_config["percentile"] <= 100.0:
        raise ValueError("pick.algorithm.depth.percentile 必须位于 0..100")
    if depth_config["max_age_ms"] < 0:
        raise ValueError("pick.algorithm.depth.max_age_ms 不得为负")
    config["pick"]["debug"]["save_root"] = str(Path(config["pick"]["debug"]["save_root"]).expanduser())
    return config
