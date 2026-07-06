"""VisionOps v3 视觉盒子设置读写。

M16 约定：
- 启动命令固定的 URL、Device ID、端口和目录只展示，不从 Web 修改。
- Web 可修改默认启动模式、状态刷新 FPS、磁盘告警阈值和服务端上传配置。
- 配置持久化到 /opt/visionops_v3/config/vision_box_settings.json，可通过环境变量覆盖。
"""

from __future__ import annotations

import json
import ipaddress
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_loader import CollectorConfig
from .response_utils import timestamp_ms

DEFAULT_PROJECT_ROOT = Path(os.environ.get("VISIONOPS_PROJECT_ROOT", "/opt/visionops_v3"))
DEFAULT_CONFIG_PATH = DEFAULT_PROJECT_ROOT / "config" / "vision_box_settings.json"
CONFIG_PATH = Path(os.environ.get("VISIONOPS_VISION_BOX_SETTINGS_FILE", str(DEFAULT_CONFIG_PATH)))


NETWORK_INTERFACES = ("eth0", "eth1")
INTERFACE_ROLES = {
    "eth0": "上位机 / 工厂网",
    "eth1": "相机 / 设备网",
}


def _run_command(args: list[str], timeout_s: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )


def _prefix_to_netmask(prefix: int | None) -> str:
    if prefix is None:
        return ""
    try:
        return str(ipaddress.IPv4Network(f"0.0.0.0/{int(prefix)}").netmask)
    except Exception:
        return ""


def _netmask_to_prefix(mask: Any) -> int:
    raw = str(mask or "").strip()
    if raw.isdigit():
        value = int(raw)
        if 0 <= value <= 32:
            return value
        raise ValueError(f"子网掩码前缀长度超出范围: {raw}")
    try:
        return int(ipaddress.IPv4Network(f"0.0.0.0/{raw}").prefixlen)
    except Exception as error:
        raise ValueError(f"子网掩码格式错误: {raw}") from error


def _validate_ipv4(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw and allow_empty:
        return ""
    try:
        return str(ipaddress.IPv4Address(raw))
    except Exception as error:
        raise ValueError(f"{field_name} 不是有效 IPv4 地址: {raw}") from error


def _read_gateway_for_interface(iface: str) -> str:
    try:
        result = _run_command(["ip", "-j", "route", "show", "default", "dev", iface], timeout_s=2.0)
        if result.returncode != 0 or not result.stdout.strip():
            return ""
        routes = json.loads(result.stdout)
        for route in routes if isinstance(routes, list) else []:
            gateway = route.get("gateway")
            if gateway:
                return str(gateway)
    except Exception:
        return ""
    return ""


def _read_interface_state(iface: str) -> dict[str, Any]:
    payload = {
        "interface": iface,
        "role": INTERFACE_ROLES.get(iface, iface),
        "exists": False,
        "state": "unknown",
        "mac": "",
        "ip": "",
        "prefix": None,
        "netmask": "",
        "gateway": "",
        "editable": True,
        "error": "",
    }
    try:
        result = _run_command(["ip", "-j", "addr", "show", "dev", iface], timeout_s=2.0)
    except Exception as error:
        payload["error"] = str(error)
        return payload
    if result.returncode != 0:
        payload["error"] = (result.stderr or result.stdout or "ip addr show failed").strip()
        return payload
    try:
        items = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        payload["error"] = f"解析 ip 输出失败: {error}"
        return payload
    if not items:
        return payload
    item = items[0]
    payload["exists"] = True
    payload["state"] = item.get("operstate") or item.get("flags", ["unknown"])[0]
    payload["mac"] = item.get("address") or ""
    for addr in item.get("addr_info", []):
        if addr.get("family") == "inet":
            payload["ip"] = addr.get("local") or ""
            payload["prefix"] = addr.get("prefixlen")
            payload["netmask"] = _prefix_to_netmask(addr.get("prefixlen"))
            break
    payload["gateway"] = _read_gateway_for_interface(iface)
    return payload


def read_dual_nic_state() -> dict[str, Any]:
    items = [_read_interface_state(iface) for iface in NETWORK_INTERFACES]
    return {
        "mode": "dual_nic_static",
        "items": items,
        "interfaces": {item["interface"]: item for item in items},
    }


def _normalize_network_interface(data: Any, iface: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = data if isinstance(data, dict) else {}
    fallback = fallback or {}
    ip_value = str(raw.get("ip", fallback.get("ip", "")) or "").strip()
    netmask_value = str(raw.get("netmask", fallback.get("netmask", "")) or "").strip()
    gateway_value = str(raw.get("gateway", fallback.get("gateway", "")) or "").strip()
    if ip_value:
        ip_value = _validate_ipv4(ip_value, f"{iface} IP")
    if netmask_value:
        prefix = _netmask_to_prefix(netmask_value)
        netmask_value = _prefix_to_netmask(prefix)
    if gateway_value:
        gateway_value = _validate_ipv4(gateway_value, f"{iface} 网关")
    return {
        "interface": iface,
        "role": INTERFACE_ROLES.get(iface, iface),
        "ip": ip_value,
        "netmask": netmask_value,
        "gateway": gateway_value,
    }


def _normalize_network(raw: Any) -> dict[str, Any]:
    live = read_dual_nic_state()["interfaces"]
    data = raw if isinstance(raw, dict) else {}
    interfaces = data.get("interfaces") if isinstance(data.get("interfaces"), dict) else {}
    return {
        "eth0": _normalize_network_interface(interfaces.get("eth0"), "eth0", live.get("eth0")),
        "eth1": _normalize_network_interface(interfaces.get("eth1"), "eth1", live.get("eth1")),
    }


def _network_differs_from_live(network: dict[str, Any]) -> bool:
    live = read_dual_nic_state()["interfaces"]
    for iface in NETWORK_INTERFACES:
        candidate = network.get(iface, {})
        current = live.get(iface, {})
        for key in ("ip", "netmask", "gateway"):
            if str(candidate.get(key, "") or "").strip() != str(current.get(key, "") or "").strip():
                return True
    return False


def _apply_one_interface(iface_config: dict[str, Any], metric: int) -> dict[str, Any]:
    iface = iface_config["interface"]
    ip_value = iface_config.get("ip") or ""
    netmask = iface_config.get("netmask") or ""
    gateway = iface_config.get("gateway") or ""
    commands: list[dict[str, Any]] = []

    def run(args: list[str], timeout_s: float = 5.0) -> None:
        result = _run_command(args, timeout_s=timeout_s)
        record = {"cmd": " ".join(args), "returncode": result.returncode}
        if result.stdout.strip():
            record["stdout"] = result.stdout.strip()
        if result.stderr.strip():
            record["stderr"] = result.stderr.strip()
        commands.append(record)
        if result.returncode != 0:
            raise RuntimeError(record.get("stderr") or record.get("stdout") or f"command failed: {' '.join(args)}")

    run(["ip", "link", "set", "dev", iface, "up"])
    if ip_value and netmask:
        prefix = _netmask_to_prefix(netmask)
        # 仅删除 IPv4 global 地址，保留 link-local/IPv6，避免过度破坏系统状态。
        run(["ip", "-4", "addr", "flush", "dev", iface, "scope", "global"])
        run(["ip", "addr", "add", f"{ip_value}/{prefix}", "dev", iface])
    if gateway:
        run(["ip", "route", "replace", "default", "via", gateway, "dev", iface, "metric", str(metric)])
    else:
        # 没有网关时删除该网口的默认路由，失败不视为错误。
        result = _run_command(["ip", "route", "del", "default", "dev", iface], timeout_s=3.0)
        record = {"cmd": f"ip route del default dev {iface}", "returncode": result.returncode, "ignored": True}
        if result.stderr.strip():
            record["stderr"] = result.stderr.strip()
        commands.append(record)
    return {"interface": iface, "commands": commands}


def apply_dual_nic_network(network: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    applied: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metrics = {"eth0": 100, "eth1": 200}
    for iface in NETWORK_INTERFACES:
        try:
            applied.append(_apply_one_interface(network[iface], metrics.get(iface, 300)))
        except Exception as error:
            errors.append({"interface": iface, "error": str(error)})
            break
    return {
        "ok": not errors,
        "applied": applied,
        "errors": errors,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 3),
        "state_after": read_dual_nic_state(),
    }


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
        "network": _normalize_network({}),
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
        "network": _normalize_network(raw.get("network") or defaults.get("network") or {}),
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
            **read_dual_nic_state(),
            "configured": settings.get("network", {}),
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
        "network": payload.get("network", before.get("network", {})),
    }, config)
    changed = candidate != before
    network_changed_live = _network_differs_from_live(candidate.get("network", {}))
    timings: dict[str, float] = {}
    network_apply_result: dict[str, Any] = {"attempted": False, "skipped": not network_changed_live, "reason": "network unchanged" if not network_changed_live else ""}
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
    if network_changed_live:
        network_apply_result = {"attempted": True, **apply_dual_nic_network(candidate.get("network", {}))}
        if not network_apply_result.get("ok"):
            raise RuntimeError(f"双网口配置应用失败: {network_apply_result.get('errors')}")
    timings["network_apply_ms"] = network_apply_result.get("duration_ms", 0)
    result = get_vision_box_settings_payload(config)
    result.update({
        "message_type": "vision_box_settings_apply_result",
        "changed": changed,
        "skipped_write": not changed,
        "network_apply": network_apply_result,
        "apply_timings_ms": timings,
    })
    return result
