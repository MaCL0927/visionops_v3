"""VisionOps v3 视觉盒子设置读写。

M16 约定：
- 启动命令固定的 URL、Device ID、端口和目录只展示，不从 Web 修改。
- Web 可修改默认启动模式、状态刷新 FPS、磁盘告警阈值和服务端上传配置。
- 配置持久化到 /opt/visionops_v3/config/vision_box_settings.json，可通过环境变量覆盖。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_loader import CollectorConfig
from .response_utils import timestamp_ms

DEFAULT_PROJECT_ROOT = Path(os.environ.get("VISIONOPS_PROJECT_ROOT", "/opt/visionops_v3"))
DEFAULT_CONFIG_PATH = DEFAULT_PROJECT_ROOT / "config" / "vision_box_settings.json"
CONFIG_PATH = Path(os.environ.get("VISIONOPS_VISION_BOX_SETTINGS_FILE", str(DEFAULT_CONFIG_PATH)))


def _clamp_number(value: Any, fallback: float, min_value: float, max_value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return max(min_value, min(max_value, number))


def _to_int(value: Any, fallback: int, min_value: int, max_value: int) -> int:
    return int(round(_clamp_number(value, fallback, min_value, max_value)))


def _status_fps_to_ms(fps: Any) -> int:
    number = _clamp_number(fps, 0.5, 0.2, 10)
    return max(100, int(round(1000 / number)))


def _status_ms_to_fps(ms: Any) -> float:
    value = _clamp_number(ms, 2000, 100, 60000)
    fps = 1000.0 / value
    # 常用值显示为整数；0.5 这类低频保留一位
    return round(fps, 2 if fps < 1 else 1)


def _default_settings(config: CollectorConfig) -> dict[str, Any]:
    return {
        "default_mode": "factory",
        "status_refresh_fps": _status_ms_to_fps(config.status_refresh_interval_ms),
        "status_refresh_interval_ms": config.status_refresh_interval_ms,
        "disk_warning_percent": 85,
        "upload": {
            "server_ip": "",
            "ssh_user": "",
            "ssh_password": "",
            "ssh_port": 22,
            "remote_dir": "/opt/visionops_uploads",
            "timeout_s": 60,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"视觉盒子配置 JSON 格式错误: {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"视觉盒子配置必须是 JSON 对象: {path}")
    return data


def _normalize_upload(upload: Any) -> dict[str, Any]:
    data = upload if isinstance(upload, dict) else {}
    server_ip = str(data.get("server_ip") or "").strip()
    ssh_user = str(data.get("ssh_user") or "").strip()
    ssh_password = str(data.get("ssh_password") or "")
    remote_dir = str(data.get("remote_dir") or "/opt/visionops_uploads").strip() or "/opt/visionops_uploads"
    return {
        "server_ip": server_ip,
        "ssh_user": ssh_user,
        "ssh_password": ssh_password,
        "ssh_port": _to_int(data.get("ssh_port"), 22, 1, 65535),
        "remote_dir": remote_dir,
        "timeout_s": _to_int(data.get("timeout_s"), 60, 5, 3600),
    }


def _normalize_settings(raw: dict[str, Any], config: CollectorConfig) -> dict[str, Any]:
    defaults = _default_settings(config)
    default_mode = str(raw.get("default_mode") or defaults["default_mode"]).strip().lower()
    if default_mode not in {"factory", "production"}:
        default_mode = "factory"
    status_fps = _clamp_number(raw.get("status_refresh_fps"), defaults["status_refresh_fps"], 0.2, 10)
    status_ms = _status_fps_to_ms(status_fps)
    return {
        "default_mode": default_mode,
        "status_refresh_fps": round(status_fps, 2 if status_fps < 1 else 1),
        "status_refresh_interval_ms": status_ms,
        "disk_warning_percent": _to_int(raw.get("disk_warning_percent"), defaults["disk_warning_percent"], 50, 99),
        "upload": _normalize_upload(raw.get("upload") or defaults["upload"]),
    }


def load_vision_box_settings(config: CollectorConfig) -> dict[str, Any]:
    raw = _read_json(CONFIG_PATH)
    return _normalize_settings(raw, config)


def _disk_payload(path: str, warning_percent: int) -> dict[str, Any]:
    target = Path(path or "/")
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        usage = shutil.disk_usage(str(existing))
        used_percent = round((usage.used / usage.total) * 100, 1) if usage.total else 0
        return {
            "path": str(target),
            "checked_path": str(existing),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": used_percent,
            "warning_percent": warning_percent,
            "warning": used_percent >= warning_percent,
        }
    except OSError as error:
        return {
            "path": str(target),
            "checked_path": str(existing),
            "warning_percent": warning_percent,
            "warning": False,
            "error": str(error),
        }


def get_vision_box_settings_payload(config: CollectorConfig) -> dict[str, Any]:
    settings = load_vision_box_settings(config)
    project_root = DEFAULT_PROJECT_ROOT
    runtime_port = _to_int(str(config.runtime_url).rsplit(":", 1)[-1] if ":" in config.runtime_url else 28081, 28081, 1, 65535)
    collector_dict = asdict(config)
    services = {
        "runtime_url": config.runtime_url,
        "gateway_url": config.gateway_url,
        "business_app_url": config.business_app_url,
        "device_id": config.device_id,
        "collector_host": config.host,
        "collector_port": config.port,
        "runtime_port": runtime_port,
    }
    paths = {
        "project_root": str(project_root),
        "models_root": config.models_root,
        "data_root": str(project_root / "data"),
        "log_root": str(project_root / "logs"),
        "config_path": str(CONFIG_PATH),
    }
    return {
        "schema_version": "1.0",
        "message_type": "vision_box_settings",
        "status": "ok",
        "timestamp_ms": timestamp_ms(),
        "settings": settings,
        "services": services,
        "paths": paths,
        "storage": {
            "project_root": _disk_payload(str(project_root), settings["disk_warning_percent"]),
            "models_root": _disk_payload(config.models_root, settings["disk_warning_percent"]),
        },
        "network": {
            "mode": "dual_nic_placeholder",
            "items": [
                {"role": "上位机 / 工厂网", "interface": "eth0", "status": "待接入"},
                {"role": "相机 / 设备网", "interface": "eth1", "status": "待接入"},
            ],
        },
        "collector_config": collector_dict,
        "config_path": str(CONFIG_PATH),
    }


def apply_vision_box_settings(config: CollectorConfig, payload: dict[str, Any]) -> dict[str, Any]:
    before = load_vision_box_settings(config)
    candidate = _normalize_settings({
        "default_mode": payload.get("default_mode", before["default_mode"]),
        "status_refresh_fps": payload.get("status_refresh_fps", before["status_refresh_fps"]),
        "disk_warning_percent": payload.get("disk_warning_percent", before["disk_warning_percent"]),
        "upload": payload.get("upload", before["upload"]),
    }, config)
    changed = candidate != before
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if changed:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        body = {
            "schema_version": "1.0",
            "updated_at_ms": timestamp_ms(),
            **candidate,
        }
        tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(CONFIG_PATH)
    timings["write_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    result = get_vision_box_settings_payload(config)
    result.update({
        "message_type": "vision_box_settings_apply_result",
        "changed": changed,
        "skipped_write": not changed,
        "apply_timings_ms": timings,
    })
    return result
