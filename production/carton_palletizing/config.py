"""Configuration loader for the standalone carton-palletizing solution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set
from urllib.parse import urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "production/carton_palletizing/config/line.yaml"


def _project_path(*parts: str) -> str:
    return str((PROJECT_ROOT.joinpath(*parts)).resolve())


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "kind": "production_line",
    "line_id": "carton_palletizing",
    "device_id": "lb3576-carton-palletizing",
    "component": "carton_palletizing_app",
    "camera_bridge": {
        "base_url": "http://127.0.0.1:18182",
        "snapshot_path": "/stream/snapshot.jpg",
        "health_path": "/health",
    },
    "runtime_recovery": {
        "stale_frame_timeout_ms": 3000,
        "failure_threshold": 3,
        "initial_backoff_ms": 200,
        "max_backoff_ms": 2000,
    },
    "runtime": {
        "url": "http://127.0.0.1:28084",
        "model_dir": _project_path("models", "carton_palletizing", "current"),
        "roi_config_path": _project_path("data", "runtime", "roi_carton_palletizing.json"),
        "device_id": "lb3576-carton-palletizing-runtime",
        "component": "rknn_runtime_carton_palletizing",
        "accepted_task_types": ["obb", "obb_detection"],
    },
    "app": {
        "listen_host": "127.0.0.1",
        "listen_port": 19210,
        "request_timeout_ms": 5000,
    },
    "collector": {
        "listen_host": "0.0.0.0",
        "listen_port": 18094,
        "device_id": "lb3576-carton-palletizing",
        "component": "collector_carton_palletizing",
        "models_root": _project_path("models", "carton_palletizing"),
        "snapshot_refresh_interval_ms": 200,
        "status_refresh_interval_ms": 2000,
        "production_inference_source": "app",
    },
    "task": {
        "task_id": "first_layer_placement",
        "algorithm": {
            "classes": {
                # Current OBB model classes are fixed: 0=box, 1=tray.
                # Class names remain as a compatibility fallback.
                "tray_class_ids": [1],
                "tray_class_names": ["tray", "pallet"],
                "box_class_ids": [0],
                "box_class_names": ["box", "carton", "carton_box"],
                "tray_min_confidence": 0.50,
                "box_min_confidence": 0.50,
            },
            "geometry": {
                "require_obb": True,
                "footprint_mode": "centered_square_by_short_edge",
                "footprint_fill_ratio": 1.0,
            },
            "tray_tracking": {
                "lock_after_first_detection": True,
                "ema_alpha": 0.35,
                "update_min_iou": 0.30,
            },
            "matching": {
                "min_iou": 0.12,
                "max_center_distance_ratio": 0.60,
                "max_orientation_diff_deg": 45.0,
                "center_inside_bonus": 0.45,
                "iou_weight": 0.35,
                "center_weight": 0.20,
                "orientation_weight": 0.10,
            },
            "temporal": {
                "occupied_confirm_frames": 2,
                "empty_confirm_frames": 5,
                "sticky_occupied": True,
            },
            "template": {
                # Start from bottom-left P3, then move clockwise in image space.
                "slot_order": ["P3", "P1", "P2", "P4"],
                # Coordinates are relative to the centered square footprint, not
                # the entire rectangular tray. Four cartons form a pinwheel square.
                "slots": [
                    {
                        "slot_id": "P1",
                        "name": "top_left_horizontal",
                        "orientation_deg": 0.0,
                        "polygon_norm": [[0.00, 0.00], [0.65, 0.00], [0.65, 0.35], [0.00, 0.35]],
                    },
                    {
                        "slot_id": "P2",
                        "name": "top_right_vertical",
                        "orientation_deg": 90.0,
                        "polygon_norm": [[0.65, 0.00], [1.00, 0.00], [1.00, 0.65], [0.65, 0.65]],
                    },
                    {
                        "slot_id": "P3",
                        "name": "bottom_left_vertical",
                        "orientation_deg": 90.0,
                        "polygon_norm": [[0.00, 0.35], [0.35, 0.35], [0.35, 1.00], [0.00, 1.00]],
                    },
                    {
                        "slot_id": "P4",
                        "name": "bottom_right_horizontal",
                        "orientation_deg": 0.0,
                        "polygon_norm": [[0.35, 0.65], [1.00, 0.65], [1.00, 1.00], [0.35, 1.00]],
                    },
                ],
            },
        },
    },
    "debug": {
        "allow_injected_runtime_result": False,
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


def _path(value: object) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _validate_algorithm(config: Dict[str, Any]) -> None:
    algorithm = config["task"]["algorithm"]
    classes = algorithm["classes"]
    for key in ("tray_class_ids", "box_class_ids"):
        classes[key] = [int(item) for item in classes.get(key, [])]
    for key in ("tray_class_names", "box_class_names"):
        classes[key] = [str(item).strip().lower() for item in classes.get(key, []) if str(item).strip()]
    if not classes["tray_class_ids"] and not classes["tray_class_names"]:
        raise ValueError("至少配置一个托盘 class_id 或 class_name")
    if not classes["box_class_ids"] and not classes["box_class_names"]:
        raise ValueError("至少配置一个纸箱 class_id 或 class_name")
    for key in ("tray_min_confidence", "box_min_confidence"):
        classes[key] = float(classes[key])
        if not 0 <= classes[key] <= 1:
            raise ValueError(f"task.algorithm.classes.{key} 必须位于 0..1")

    geometry = algorithm.get("geometry", {})
    geometry["require_obb"] = bool(geometry.get("require_obb", True))
    geometry["footprint_mode"] = str(
        geometry.get("footprint_mode", "centered_square_by_short_edge")
    ).strip().lower()
    if geometry["footprint_mode"] not in {"centered_square_by_short_edge", "full_tray"}:
        raise ValueError("geometry.footprint_mode 必须为 centered_square_by_short_edge 或 full_tray")
    geometry["footprint_fill_ratio"] = float(geometry.get("footprint_fill_ratio", 1.0))
    if not 0.1 <= geometry["footprint_fill_ratio"] <= 1.0:
        raise ValueError("geometry.footprint_fill_ratio 必须位于 0.1..1.0")
    algorithm["geometry"] = geometry

    matching = algorithm["matching"]
    for key in ("min_iou", "max_center_distance_ratio", "center_inside_bonus", "iou_weight", "center_weight", "orientation_weight"):
        matching[key] = float(matching[key])
    matching["max_orientation_diff_deg"] = float(matching["max_orientation_diff_deg"])
    if not 0 <= matching["min_iou"] <= 1:
        raise ValueError("matching.min_iou 必须位于 0..1")
    if matching["max_center_distance_ratio"] <= 0:
        raise ValueError("matching.max_center_distance_ratio 必须大于 0")
    if not 0 < matching["max_orientation_diff_deg"] <= 90:
        raise ValueError("matching.max_orientation_diff_deg 必须位于 (0, 90]")

    tracking = algorithm["tray_tracking"]
    tracking["ema_alpha"] = float(tracking["ema_alpha"])
    tracking["update_min_iou"] = float(tracking["update_min_iou"])
    if not 0 < tracking["ema_alpha"] <= 1:
        raise ValueError("tray_tracking.ema_alpha 必须位于 (0, 1]")
    if not 0 <= tracking["update_min_iou"] <= 1:
        raise ValueError("tray_tracking.update_min_iou 必须位于 0..1")

    temporal = algorithm["temporal"]
    for key in ("occupied_confirm_frames", "empty_confirm_frames"):
        temporal[key] = int(temporal[key])
        if temporal[key] <= 0:
            raise ValueError(f"task.algorithm.temporal.{key} 必须大于 0")

    template = algorithm["template"]
    slots = template.get("slots")
    if not isinstance(slots, list) or len(slots) != 4:
        raise ValueError("第一阶段固定要求 template.slots 恰好包含 4 个摆放区域")
    seen = set()  # type: Set[str]
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise ValueError("template.slots 每项必须是对象")
        slot_id = str(slot.get("slot_id") or "").strip()
        if not slot_id or slot_id in seen:
            raise ValueError("template.slots.slot_id 必须非空且唯一")
        seen.add(slot_id)
        polygon = slot.get("polygon_norm")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"slot {slot_id} 的 polygon_norm 至少包含 3 个点")
        normalized = []  # type: List[List[float]]
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"slot {slot_id} 的 polygon_norm 点格式错误")
            normalized.append([float(point[0]), float(point[1])])
        slot["polygon_norm"] = normalized
        slot["orientation_deg"] = float(slot.get("orientation_deg", 0.0))
    order = [str(item) for item in template.get("slot_order", [])]
    if set(order) != seen or len(order) != 4:
        raise ValueError("template.slot_order 必须完整列出 4 个 slot_id")
    template["slot_order"] = order


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    source = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if source.is_file():
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"无法读取纸箱摆放配置 {source}: {error}") from error
        if not isinstance(document, Mapping):
            raise ValueError("纸箱摆放配置顶层必须是对象")
        if document.get("kind") not in {None, "production_line"}:
            raise ValueError("配置 kind 必须为 production_line")
        config = _merge(config, document)
    elif path:
        raise ValueError(f"配置文件不存在: {source}")

    config["camera_bridge"]["base_url"] = _url(config["camera_bridge"]["base_url"], "camera_bridge.base_url")
    config["runtime"]["url"] = _url(config["runtime"]["url"], "runtime.url")
    config["runtime"]["model_dir"] = _path(config["runtime"]["model_dir"])
    config["runtime"]["roi_config_path"] = _path(config["runtime"]["roi_config_path"])
    config["runtime"]["accepted_task_types"] = [
        str(item).strip().lower() for item in config["runtime"].get("accepted_task_types", [])
    ]

    config["app"]["listen_port"] = _port(config["app"]["listen_port"], "app.listen_port")
    config["collector"]["listen_port"] = _port(config["collector"]["listen_port"], "collector.listen_port")
    runtime_port = urlparse(config["runtime"]["url"]).port or 80
    ports = [runtime_port, config["app"]["listen_port"], config["collector"]["listen_port"]]
    if len(ports) != len(set(ports)):
        raise ValueError("Runtime、业务应用和 Collector 端口必须互不相同")

    config["app"]["request_timeout_ms"] = int(config["app"]["request_timeout_ms"])
    if config["app"]["request_timeout_ms"] <= 0:
        raise ValueError("app.request_timeout_ms 必须大于 0")
    for key in ("snapshot_refresh_interval_ms", "status_refresh_interval_ms"):
        config["collector"][key] = int(config["collector"][key])
        if config["collector"][key] < 100:
            raise ValueError(f"collector.{key} 不得小于 100")
    if config["collector"].get("production_inference_source") != "app":
        raise ValueError("纸箱摆放第一阶段必须使用 collector.production_inference_source=app")

    _validate_algorithm(config)
    return config
