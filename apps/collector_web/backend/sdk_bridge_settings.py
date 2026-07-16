"""Unified SDK Bridge settings for Orbbec 336L and HP60C.

Both bridges run on dedicated ports. Saving camera_model updates the shared active
camera selection and restarts the current Runtime service, so every Web page
continues to use /api/runtime/snapshot.jpg while the underlying bridge changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from edge.camera_bridge.camera_selection import (
    CAMERA_SPECS,
    active_camera_spec,
    normalize_camera_model,
    read_camera_selection,
    write_camera_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_RE = re.compile(r"^[a-zA-Z0-9_-]+:(\d+)x(\d+)@(\d+)$")

CAMERA_CONFIG: dict[str, dict[str, Any]] = {
    "orbbec336l": {
        "prefix": "VISIONOPS_ORBBEC336L_",
        "env_path": Path(os.environ.get(
            "VISIONOPS_ORBBEC336L_BRIDGE_ENV",
            "/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env",
        )),
        "service": os.environ.get("VISIONOPS_ORBBEC336L_SERVICE", "visionops-orbbec336l-bridge.service"),
        "defaults": {
            "HTTP_HOST": "0.0.0.0", "HTTP_PORT": "18182", "COLOR_WIDTH": "640", "COLOR_HEIGHT": "480",
            "DEPTH_WIDTH": "640", "DEPTH_HEIGHT": "480", "FPS": "30", "MJPEG_FPS": "10",
            "JPEG_QUALITY": "85", "FLIP_VERTICAL": "false", "FLIP_HORIZONTAL": "false",
            "DEPTH_UNIT": "mm", "SERIAL": "",
        },
    },
    "hp60c": {
        "prefix": "VISIONOPS_HP60C_",
        "env_path": Path(os.environ.get(
            "VISIONOPS_HP60C_BRIDGE_ENV",
            "/opt/visionops_v3/edge/camera_bridge/hp60c_bridge/hp60c_sdk_bridge.env",
        )),
        "service": os.environ.get("VISIONOPS_HP60C_SERVICE", "visionops-hp60c-sdk-bridge.service"),
        "defaults": {
            "HTTP_HOST": "0.0.0.0", "HTTP_PORT": "18181", "COLOR_WIDTH": "640", "COLOR_HEIGHT": "480",
            "DEPTH_WIDTH": "640", "DEPTH_HEIGHT": "480", "FPS": "30", "MJPEG_FPS": "10",
            "JPEG_QUALITY": "85", "FLIP_VERTICAL": "true", "FLIP_HORIZONTAL": "false",
            "DEPTH_UNIT": "mm", "RGB_SOURCE": "auto", "RGB_ORDER": "bgr", "CONFIG": "",
            "FX": "0", "FY": "0", "CX": "0", "CY": "0",
        },
    },
}


def read_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    lines: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values, lines
    for line in text.splitlines():
        lines.append(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values, lines


def write_env_file(path: Path, existing_lines: list[str], updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    output: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                output.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        output.append(line)
    missing = [key for key in updates if key not in seen]
    if missing:
        if output and output[-1].strip():
            output.append("")
        output.append("# Updated by VisionOps Collector camera settings")
        output.extend(f"{key}={updates[key]}" for key in missing)
    body = "\n".join(output).rstrip() + "\n"

    def direct_write(directory: Path) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
            os.chmod(tmp_name, 0o664)
            os.replace(tmp_name, path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass

    try:
        direct_write(path.parent)
        return
    except PermissionError:
        pass
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        command = ["install", "-m", "0664", tmp_name, str(path)] if os.geteuid() == 0 else ["sudo", "-n", "install", "-m", "0664", tmp_name, str(path)]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=8, check=False)
        if proc.returncode != 0:
            raise PermissionError(proc.stderr.strip() or f"无法写入 {path}")
    finally:
        try: os.unlink(tmp_name)
        except OSError: pass


def parse_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    return str(value).strip().lower() in {"1", "true", "yes", "on", "开启", "开"}


def clamp_int(value: Any, fallback: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    return max(lo, min(hi, number))


def camera_config(camera_model: str) -> dict[str, Any]:
    model = normalize_camera_model(camera_model, "")
    if model not in CAMERA_CONFIG:
        raise ValueError(f"不支持的相机型号: {camera_model}")
    return CAMERA_CONFIG[model]


def env_value(values: dict[str, str], cfg: dict[str, Any], suffix: str) -> str:
    return values.get(cfg["prefix"] + suffix, cfg["defaults"].get(suffix, ""))


def profile_id(model: str, width: int, height: int, fps: int) -> str:
    return f"{model}:{int(width)}x{int(height)}@{int(fps)}"


def parse_profile(profile: str, expected_model: str) -> dict[str, int]:
    match = PROFILE_RE.match(str(profile or "").strip())
    if not match:
        raise ValueError(f"profile 格式非法: {profile!r}")
    prefix = str(profile).split(":", 1)[0]
    if normalize_camera_model(prefix, "") != expected_model:
        raise ValueError(f"profile {profile!r} 不属于 {expected_model}")
    return {"width": int(match.group(1)), "height": int(match.group(2)), "fps": int(match.group(3))}


def make_profile(model: str, width: int, height: int, fps: int, sensor: str,
                 formats: list[str] | None = None, source: str = "env") -> dict[str, Any]:
    prefix = "RGB" if sensor == "color" else "Depth"
    suffix = f" ({'/'.join(formats or [])})" if formats else ""
    return {
        "id": profile_id(model, width, height, fps), "sensor": sensor,
        "width": int(width), "height": int(height), "fps": int(fps),
        "formats": formats or [], "label": f"{prefix} {width}×{height} @ {fps} FPS{suffix}",
        "source": source,
    }


def current_profiles_from_env(model: str, values: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = camera_config(model)
    fps = clamp_int(env_value(values, cfg, "FPS"), 30, 1, 120)
    color_w = clamp_int(env_value(values, cfg, "COLOR_WIDTH"), 640, 1, 8192)
    color_h = clamp_int(env_value(values, cfg, "COLOR_HEIGHT"), 480, 1, 8192)
    depth_w = clamp_int(env_value(values, cfg, "DEPTH_WIDTH"), color_w, 1, 8192)
    depth_h = clamp_int(env_value(values, cfg, "DEPTH_HEIGHT"), color_h, 1, 8192)
    return (
        make_profile(model, color_w, color_h, fps, "color", [], "env_current"),
        make_profile(model, depth_w, depth_h, fps, "depth", ["Y16"], "env_current"),
    )


def bridge_base_url(model: str, values: dict[str, str]) -> str:
    cfg = camera_config(model)
    # HTTP_HOST=0.0.0.0 is a listen address, not a client target.
    host = env_value(values, cfg, "HTTP_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = clamp_int(env_value(values, cfg, "HTTP_PORT"), int(CAMERA_SPECS[model]["public_port"]), 1, 65535)
    return f"http://{host}:{port}"


def fetch_json(url: str, timeout_s: float = 2.5) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - local bridge URL
        body = response.read(1024 * 1024)
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层不是对象")
    return data


def normalize_bridge_profiles(model: str, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else payload
    def normalize(items: Any, sensor: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, int, int], set[str]] = {}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            width = clamp_int(item.get("width"), 0, 0, 8192)
            height = clamp_int(item.get("height"), 0, 0, 8192)
            fps = clamp_int(item.get("fps"), 0, 0, 240)
            if not width or not height or not fps:
                continue
            formats = item.get("formats") if isinstance(item.get("formats"), list) else [item.get("format")]
            grouped.setdefault((width, height, fps), set()).update(str(v) for v in formats if v)
        return sorted(
            [make_profile(model, w, h, fps, sensor, sorted(fmts), "bridge_api") for (w, h, fps), fmts in grouped.items()],
            key=lambda p: (-p["width"] * p["height"], -p["fps"], p["id"]),
        )
    return normalize(profiles.get("color", []) if isinstance(profiles, dict) else [], "color"), normalize(profiles.get("depth", []) if isinstance(profiles, dict) else [], "depth")


def collect_profiles(model: str, values: dict[str, str]) -> dict[str, Any]:
    profile_url = bridge_base_url(model, values) + CAMERA_SPECS[model]["profiles_path"]
    fallback_color, fallback_depth = current_profiles_from_env(model, values)
    try:
        color, depth = normalize_bridge_profiles(model, fetch_json(profile_url))
        if color and depth:
            return {"source": "bridge_api", "profile_url": profile_url, "color": color, "depth": depth, "warning": None}
        warning = "Bridge profile API 返回空列表，已回退到 env"
    except (OSError, URLError, HTTPError, ValueError, json.JSONDecodeError) as error:
        warning = f"无法从 Bridge 枚举 profile，已回退到 env: {error}"
    return {"source": "env_current_fallback", "profile_url": profile_url, "color": [fallback_color], "depth": [fallback_depth], "warning": warning}


def command_result(args: list[str], timeout_s: float = 15.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout_s, check=False)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "args": args, "elapsed_ms": round((time.monotonic()-started)*1000, 3)}
    except FileNotFoundError as error:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": str(error), "args": args}
    except subprocess.TimeoutExpired as error:
        return {"ok": False, "returncode": 124, "stdout": error.stdout or "", "stderr": error.stderr or "timeout", "args": args}


def service_status(service: str) -> dict[str, Any]:
    active = command_result(["systemctl", "is-active", service], 3)
    enabled = command_result(["systemctl", "is-enabled", service], 3)
    return {"name": service, "active": active.get("stdout") or "unknown", "enabled": enabled.get("stdout") or "unknown", "active_ok": active.get("ok", False)}


def restart_service(service: str) -> dict[str, Any]:
    result = command_result(["systemctl", "restart", service], 18)
    if not result["ok"] and os.geteuid() != 0:
        fallback = command_result(["sudo", "-n", "systemctl", "restart", service], 18)
        fallback["fallback_from"] = result
        result = fallback
    return result


def wait_bridge_health(model: str, values: dict[str, str], timeout_s: float = 12.0) -> dict[str, Any]:
    """Wait until the selected Bridge is reachable and has fresh RGB/depth frames."""
    url = bridge_base_url(model, values) + CAMERA_SPECS[model]["health_path"]
    deadline = time.monotonic() + timeout_s
    last_error = ""
    last_response: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            data = fetch_json(url, 1.0)
            last_response = data
            if data.get("camera_connected") is True:
                return {"ok": True, "url": url, "response": data}
            last_error = str(data.get("error") or data.get("camera_state") or "camera has no fresh RGB/depth frame")
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
        time.sleep(0.25)
    return {"ok": False, "url": url, "error": last_error or "timeout", "response": last_response}


def current_settings(model: str, values: dict[str, str]) -> dict[str, Any]:
    cfg = camera_config(model)
    color, depth = current_profiles_from_env(model, values)
    result = {
        "camera_model": model, "rgb_profile": color["id"], "depth_profile": depth["id"],
        "display_fps": clamp_int(env_value(values, cfg, "MJPEG_FPS"), 10, 1, 30),
        "camera_jpeg_quality": clamp_int(env_value(values, cfg, "JPEG_QUALITY"), 85, 10, 100),
        "flip_vertical": str(parse_bool(env_value(values, cfg, "FLIP_VERTICAL"), model == "hp60c")).lower(),
        "flip_horizontal": str(parse_bool(env_value(values, cfg, "FLIP_HORIZONTAL"), False)).lower(),
        "depth_unit": env_value(values, cfg, "DEPTH_UNIT") or "mm",
        "bridge_url": bridge_base_url(model, values),
        "orbbec_serial": env_value(values, cfg, "SERIAL") if model == "orbbec336l" else "",
        "rgb_source_preference": env_value(values, cfg, "RGB_SOURCE") if model == "hp60c" else "auto",
        "rgb_order": env_value(values, cfg, "RGB_ORDER") if model == "hp60c" else "auto",
        "hp60c_config_path": env_value(values, cfg, "CONFIG") if model == "hp60c" else "",
        "hp60c_fx": env_value(values, cfg, "FX") if model == "hp60c" else "",
        "hp60c_fy": env_value(values, cfg, "FY") if model == "hp60c" else "",
        "hp60c_cx": env_value(values, cfg, "CX") if model == "hp60c" else "",
        "hp60c_cy": env_value(values, cfg, "CY") if model == "hp60c" else "",
        "profile_edit_mode": "vendor_config_file" if model == "hp60c" else "sdk_profile",
    }
    return result


def get_sdk_bridge_settings_payload(camera_model: str | None = None) -> dict[str, Any]:
    selection = read_camera_selection()
    active = normalize_camera_model(selection.get("active_camera"), "orbbec336l")
    model = normalize_camera_model(camera_model, active)
    cfg = camera_config(model)
    path: Path = cfg["env_path"]
    values, _ = read_env_file(path)
    return {
        "schema_version": "2.0", "message_type": "sdk_bridge_settings",
        "active_camera": active, "camera_model": model,
        "available_cameras": [
            {"camera_model": key, "display_name": CAMERA_SPECS[key]["display_name"], "base_url": CAMERA_SPECS[key]["base_url"], "service": CAMERA_CONFIG[key]["service"]}
            for key in ("orbbec336l", "hp60c")
        ],
        "selection_path": selection["path"], "env_path": str(path), "env_exists": path.exists(),
        "service": service_status(cfg["service"]), "settings": current_settings(model, values),
        "profiles": collect_profiles(model, values),
    }


def _changed(values: dict[str, str], updates: dict[str, str]) -> list[str]:
    return [key for key, value in updates.items() if str(values.get(key, "")).strip().lower() != str(value).strip().lower()]


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _runtime_service_candidates() -> list[str]:
    explicit = [item.strip() for item in os.environ.get("VISIONOPS_COLLECTOR_RUNTIME_SERVICE", "").split(",") if item.strip()]
    return _ordered_unique(explicit + [
        "visionops-v3-runtime-partition.service",
        "visionops-v3-runtime-tube.service",
        "visionops-v3-runtime-pick.service",
        "visionops-v3-carton-palletizing-runtime.service",
    ])


def _camera_dependent_service_candidates() -> list[str]:
    explicit = [item.strip() for item in os.environ.get("VISIONOPS_COLLECTOR_CAMERA_DEPENDENT_SERVICES", "").split(",") if item.strip()]
    return _ordered_unique(explicit + [
        "visionops-v3-robot-gateway.service",
        "visionops-v3-ws-pick.service",
        "visionops-v3-carton-palletizing-app.service",
    ])


def restart_camera_consumers() -> list[dict[str, Any]]:
    """Restart every active process that reads the selected Bridge at process start."""
    results: list[dict[str, Any]] = []
    # Runtime first, then RGB-D business services so consumers see the new Runtime/Bridge.
    for role, services in (
        ("runtime", _runtime_service_candidates()),
        ("business", _camera_dependent_service_candidates()),
    ):
        for service in services:
            active = command_result(["systemctl", "is-active", "--quiet", service], 3)
            if not active["ok"]:
                continue
            result = restart_service(service)
            result["service"] = service
            result["role"] = role
            results.append(result)
    return results


def apply_sdk_bridge_settings(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    model = normalize_camera_model(payload.get("camera_model"), "")
    if model not in CAMERA_CONFIG:
        raise ValueError("camera_model 必须为 orbbec336l 或 hp60c")
    cfg = camera_config(model)
    path: Path = cfg["env_path"]
    values, lines = read_env_file(path)
    color = parse_profile(str(payload.get("rgb_profile") or ""), model)
    depth = parse_profile(str(payload.get("depth_profile") or ""), model)
    if color["fps"] != depth["fps"]:
        raise ValueError("RGB 与 Depth FPS 必须一致")
    prefix = cfg["prefix"]
    updates = {
        prefix + "COLOR_WIDTH": str(color["width"]),
        prefix + "COLOR_HEIGHT": str(color["height"]),
        prefix + "DEPTH_WIDTH": str(depth["width"]),
        prefix + "DEPTH_HEIGHT": str(depth["height"]),
        prefix + "FPS": str(color["fps"]),
        prefix + "MJPEG_FPS": str(clamp_int(payload.get("display_fps"), 10, 1, 30)),
        prefix + "JPEG_QUALITY": str(clamp_int(payload.get("camera_jpeg_quality"), 85, 10, 100)),
        prefix + "FLIP_VERTICAL": str(parse_bool(payload.get("flip_vertical"), model == "hp60c")).lower(),
        prefix + "FLIP_HORIZONTAL": str(parse_bool(payload.get("flip_horizontal"), False)).lower(),
        prefix + "DEPTH_UNIT": str(payload.get("depth_unit") or "mm"),
    }
    if model == "orbbec336l":
        updates[prefix + "SERIAL"] = str(payload.get("orbbec_serial") or "").strip()
    else:
        source = str(payload.get("rgb_source_preference") or "auto").strip().lower()
        if source not in {"auto", "mjpeg", "rgb", "yuyv"}:
            raise ValueError("HP60C RGB source 必须为 auto/mjpeg/rgb/yuyv")
        order = str(payload.get("rgb_order") or "bgr").strip().lower()
        if order not in {"bgr", "rgb"}:
            raise ValueError("HP60C RGB order 必须为 bgr/rgb")
        updates[prefix + "RGB_SOURCE"] = source
        updates[prefix + "RGB_ORDER"] = order
        config_path = str(payload.get("hp60c_config_path") or env_value(values, cfg, "CONFIG")).strip()
        if config_path:
            updates[prefix + "CONFIG"] = config_path
        for suffix, key in (("FX", "hp60c_fx"), ("FY", "hp60c_fy"), ("CX", "hp60c_cx"), ("CY", "hp60c_cy")):
            raw = payload.get(key, env_value(values, cfg, suffix))
            try:
                value = float(raw or 0)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} 必须为数字") from error
            if suffix in {"FX", "FY"} and value < 0:
                raise ValueError(f"{key} 不能小于 0")
            updates[prefix + suffix] = f"{value:.9g}"

    changed_keys = _changed(values, updates)
    if changed_keys:
        write_env_file(path, lines, updates)

    selection_before = active_camera_spec()["camera_model"]
    switch_changed = selection_before != model
    bridge_restart: dict[str, Any] | None = None
    bridge_health: dict[str, Any] | None = None

    # Verify the selected Bridge before publishing the camera selection. This prevents
    # a failed HP60C/336L startup from switching every Web page to a dead source.
    if changed_keys or switch_changed:
        bridge_restart = restart_service(cfg["service"])
        if not bridge_restart.get("ok"):
            raise RuntimeError(
                f"重启 {cfg['service']} 失败: "
                f"{bridge_restart.get('stderr') or bridge_restart.get('stdout') or 'unknown error'}"
            )
        updated_values, _ = read_env_file(path)
        bridge_health = wait_bridge_health(model, updated_values)
        if not bridge_health.get("ok"):
            raise RuntimeError(
                f"{CAMERA_SPECS[model]['display_name']} Bridge 重启后没有新 RGB/Depth 帧: "
                f"{bridge_health.get('error') or 'health timeout'}"
            )

    selection = write_camera_selection(model) if switch_changed else read_camera_selection()
    consumer_restarts: list[dict[str, Any]] = []
    if switch_changed or changed_keys:
        consumer_restarts = restart_camera_consumers()
        failed = [item for item in consumer_restarts if not item.get("ok")]
        if failed:
            details = "; ".join(
                f"{item.get('service')}: {item.get('stderr') or item.get('stdout') or item.get('returncode')}"
                for item in failed
            )
            raise RuntimeError(f"相机已应用，但重启图像消费者失败: {details}")

    response = get_sdk_bridge_settings_payload(model)
    response.update({
        "changed": bool(changed_keys or switch_changed),
        "changed_env_keys": changed_keys,
        "camera_switched": switch_changed,
        "active_camera": model,
        "selection": selection,
        "bridge_restart": bridge_restart,
        "bridge_health": bridge_health,
        "consumer_restarts": consumer_restarts,
        # Backward-compatible response field used by earlier frontend/tests.
        "runtime_restarts": [item for item in consumer_restarts if item.get("role") == "runtime"],
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
    })
    return response

