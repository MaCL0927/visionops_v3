"""SDK Bridge 设置 API：读取 / 写入 Orbbec 336L env，并触发受限 systemd 重启。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

ORBBEC_ENV_DEFAULT = Path("/opt/visionops_v3/edge/camera_bridge/orbbec336l_bridge/orbbec336l_bridge.env")
ORBBEC_SERVICE_NAME_DEFAULT = "visionops-orbbec336l-bridge.service"
PROFILE_RE = re.compile(r"^orbbec:(\d+)x(\d+)@(\d+)$")


def orbbec_env_path() -> Path:
    return Path(os.environ.get("VISIONOPS_ORBBEC336L_BRIDGE_ENV", str(ORBBEC_ENV_DEFAULT)))


def orbbec_service_name() -> str:
    return os.environ.get("VISIONOPS_ORBBEC336L_SERVICE", ORBBEC_SERVICE_NAME_DEFAULT)


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
        key = key.strip()
        if not key or any(c.isspace() for c in key):
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values, lines


def write_env_file(path: Path, existing_lines: list[str], updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    out: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    missing = [key for key in updates if key not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Updated by VisionOps Collector Web settings API")
        for key in missing:
            out.append(f"{key}={updates[key]}")
    body = "\n".join(out).rstrip() + "\n"
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(body)
        try:
            os.chmod(tmp_name, path.stat().st_mode & 0o777)
        except OSError:
            os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def parse_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "开启", "开"}:
        return True
    if s in {"0", "false", "no", "off", "关闭", "关"}:
        return False
    return fallback


def clamp_int(value: Any, fallback: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    return max(lo, min(hi, number))


def parse_orbbec_profile(profile_id: str) -> dict[str, int]:
    match = PROFILE_RE.match(str(profile_id or "").strip())
    if not match:
        raise ValueError(f"profile 格式非法: {profile_id!r}，期望 orbbec:WIDTHxHEIGHT@FPS")
    return {
        "width": int(match.group(1)),
        "height": int(match.group(2)),
        "fps": int(match.group(3)),
    }


def profile_id(width: int, height: int, fps: int) -> str:
    return f"orbbec:{int(width)}x{int(height)}@{int(fps)}"


def profile_label(width: int, height: int, fps: int, sensor: str, formats: list[str] | None = None) -> str:
    prefix = "RGB" if sensor == "color" else "Depth"
    suffix = f" ({'/'.join(formats)})" if formats else ""
    return f"{prefix} {width}×{height} @ {fps} FPS{suffix}"


def make_profile(width: int, height: int, fps: int, sensor: str, formats: list[str] | None = None, source: str = "env") -> dict[str, Any]:
    return {
        "id": profile_id(width, height, fps),
        "sensor": sensor,
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "formats": formats or [],
        "label": profile_label(width, height, fps, sensor, formats),
        "source": source,
    }


def current_profiles_from_env(values: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    fps = clamp_int(values.get("VISIONOPS_ORBBEC336L_FPS"), 30, 1, 120)
    color_w = clamp_int(values.get("VISIONOPS_ORBBEC336L_COLOR_WIDTH"), 1280, 0, 8192)
    color_h = clamp_int(values.get("VISIONOPS_ORBBEC336L_COLOR_HEIGHT"), 720, 0, 8192)
    depth_w = clamp_int(values.get("VISIONOPS_ORBBEC336L_DEPTH_WIDTH"), color_w or 1280, 0, 8192)
    depth_h = clamp_int(values.get("VISIONOPS_ORBBEC336L_DEPTH_HEIGHT"), color_h or 720, 0, 8192)
    return (
        make_profile(color_w, color_h, fps, "color", [], "env_current"),
        make_profile(depth_w, depth_h, fps, "depth", ["Y16"], "env_current"),
    )


def normalize_bridge_profiles(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else payload
    color_raw = profiles.get("color", []) if isinstance(profiles, dict) else []
    depth_raw = profiles.get("depth", []) if isinstance(profiles, dict) else []

    def normalize(items: list[Any], sensor: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, int, int], set[str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            width = clamp_int(item.get("width"), 0, 0, 8192)
            height = clamp_int(item.get("height"), 0, 0, 8192)
            fps = clamp_int(item.get("fps"), 0, 0, 240)
            if width <= 0 or height <= 0 or fps <= 0:
                continue
            formats = item.get("formats")
            if isinstance(formats, list):
                fmt_values = {str(v) for v in formats if str(v)}
            elif item.get("format"):
                fmt_values = {str(item.get("format"))}
            else:
                fmt_values = set()
            grouped.setdefault((width, height, fps), set()).update(fmt_values)
        out = [make_profile(w, h, fps, sensor, sorted(fmts), "bridge_api") for (w, h, fps), fmts in grouped.items()]
        return sorted(out, key=lambda p: (-p["width"] * p["height"], -p["fps"], p["id"]))

    return normalize(color_raw, "color"), normalize(depth_raw, "depth")


def bridge_base_url(values: dict[str, str]) -> str:
    host = values.get("VISIONOPS_ORBBEC336L_HTTP_HOST", "127.0.0.1") or "127.0.0.1"
    port = clamp_int(values.get("VISIONOPS_ORBBEC336L_HTTP_PORT"), 18182, 1, 65535)
    return f"http://{host}:{port}"


def fetch_json(url: str, timeout_s: float = 2.5) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - local bridge URL only
        content_type = response.headers.get("Content-Type", "")
        body = response.read(1024 * 1024)
    if "json" not in content_type.lower():
        raise ValueError(f"非 JSON 响应: {content_type}")
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层不是对象")
    return data


def collect_orbbec_profiles(values: dict[str, str], timeout_s: float = 2.5) -> dict[str, Any]:
    base = bridge_base_url(values)
    profile_url = f"{base}/stream/profiles"
    current_color, current_depth = current_profiles_from_env(values)
    try:
        payload = fetch_json(profile_url, timeout_s=timeout_s)
        color, depth = normalize_bridge_profiles(payload)
        if color and depth:
            return {
                "source": "bridge_api",
                "profile_url": profile_url,
                "color": color,
                "depth": depth,
                "warning": None,
            }
        warning = "Bridge profile API 返回空列表，已回退到当前 env profile"
    except (OSError, URLError, HTTPError, ValueError, json.JSONDecodeError) as error:
        warning = f"无法从 Bridge 枚举 SDK profile，已回退到当前 env profile: {error}"
    return {
        "source": "env_current_fallback",
        "profile_url": profile_url,
        "color": [current_color],
        "depth": [current_depth],
        "warning": warning,
    }


def command_result(args: list[str], timeout_s: float = 8.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout_s, check=False)
    except FileNotFoundError as error:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": str(error), "args": args, "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
    except subprocess.TimeoutExpired as error:
        return {"ok": False, "returncode": 124, "stdout": error.stdout or "", "stderr": error.stderr or "timeout", "args": args, "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "args": args, "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}


def systemd_status() -> dict[str, Any]:
    service = orbbec_service_name()
    active = command_result(["systemctl", "is-active", service], timeout_s=3)
    enabled = command_result(["systemctl", "is-enabled", service], timeout_s=3)
    return {
        "name": service,
        "active": active.get("stdout") or "unknown",
        "enabled": enabled.get("stdout") or "unknown",
        "active_ok": active.get("ok", False),
    }


def restart_orbbec_service() -> dict[str, Any]:
    service = orbbec_service_name()
    result = command_result(["systemctl", "restart", service], timeout_s=12)
    if not result["ok"] and os.geteuid() != 0:
        sudo_result = command_result(["sudo", "-n", "systemctl", "restart", service], timeout_s=12)
        sudo_result["fallback_from"] = result
        result = sudo_result
    return result


def wait_bridge_health(values: dict[str, str], timeout_s: float = 4.0) -> dict[str, Any]:
    base = bridge_base_url(values)
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            data = fetch_json(f"{base}/health", timeout_s=1.0)
            return {"ok": True, "url": f"{base}/health", "response": data}
        except Exception as error:  # noqa: BLE001 - convert to diagnostic JSON
            last_error = str(error)
            time.sleep(0.2)
    return {"ok": False, "url": f"{base}/health", "error": last_error or "timeout"}


def current_settings(values: dict[str, str]) -> dict[str, Any]:
    color_profile, depth_profile = current_profiles_from_env(values)
    return {
        "camera_model": "orbbec336l",
        "rgb_profile": color_profile["id"],
        "depth_profile": depth_profile["id"],
        "display_fps": clamp_int(values.get("VISIONOPS_ORBBEC336L_MJPEG_FPS"), 10, 1, 30),
        "camera_jpeg_quality": clamp_int(values.get("VISIONOPS_ORBBEC336L_JPEG_QUALITY"), 85, 10, 100),
        "flip_vertical": str(parse_bool(values.get("VISIONOPS_ORBBEC336L_FLIP_VERTICAL"), False)).lower(),
        "flip_horizontal": str(parse_bool(values.get("VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL"), False)).lower(),
        "depth_unit": values.get("VISIONOPS_ORBBEC336L_DEPTH_UNIT", "mm") or "mm",
        "orbbec_serial": values.get("VISIONOPS_ORBBEC336L_SERIAL", ""),
        "bridge_url": bridge_base_url(values),
    }


def get_orbbec_settings_payload() -> dict[str, Any]:
    path = orbbec_env_path()
    values, _lines = read_env_file(path)
    profiles = collect_orbbec_profiles(values)
    return {
        "schema_version": "1.0",
        "message_type": "sdk_bridge_settings",
        "camera_model": "orbbec336l",
        "env_path": str(path),
        "env_exists": path.exists(),
        "service": systemd_status(),
        "settings": current_settings(values),
        "profiles": profiles,
    }


def validate_profile_against(profile: dict[str, int], candidates: list[dict[str, Any]]) -> bool:
    wanted = profile_id(profile["width"], profile["height"], profile["fps"])
    return any(item.get("id") == wanted for item in candidates)


BOOL_ENV_KEYS = {
    "VISIONOPS_ORBBEC336L_FLIP_VERTICAL",
    "VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL",
}
INT_ENV_KEYS = {
    "VISIONOPS_ORBBEC336L_COLOR_WIDTH",
    "VISIONOPS_ORBBEC336L_COLOR_HEIGHT",
    "VISIONOPS_ORBBEC336L_DEPTH_WIDTH",
    "VISIONOPS_ORBBEC336L_DEPTH_HEIGHT",
    "VISIONOPS_ORBBEC336L_FPS",
    "VISIONOPS_ORBBEC336L_JPEG_QUALITY",
    "VISIONOPS_ORBBEC336L_MJPEG_FPS",
}


def env_value_matches(values: dict[str, str], key: str, wanted: str) -> bool:
    if key not in values:
        return False
    current = values.get(key, "")
    if key in BOOL_ENV_KEYS:
        return parse_bool(current) == parse_bool(wanted)
    if key in INT_ENV_KEYS:
        try:
            return int(str(current).strip()) == int(str(wanted).strip())
        except (TypeError, ValueError):
            return str(current).strip() == str(wanted).strip()
    return str(current).strip() == str(wanted).strip()


def changed_env_keys(values: dict[str, str], updates: dict[str, str]) -> list[str]:
    return [key for key, wanted in updates.items() if not env_value_matches(values, key, wanted)]


def profiles_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Use profiles previously enumerated by GET /settings to avoid re-enumerating SDK during POST."""
    profiles = payload.get("known_profiles")
    if not isinstance(profiles, dict):
        return None
    color = profiles.get("color")
    depth = profiles.get("depth")
    if not isinstance(color, list) or not isinstance(depth, list) or not color or not depth:
        return None

    def clean(items: list[Any], sensor: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            width = clamp_int(item.get("width"), 0, 0, 8192)
            height = clamp_int(item.get("height"), 0, 0, 8192)
            fps = clamp_int(item.get("fps"), 0, 0, 240)
            if width <= 0 or height <= 0 or fps <= 0:
                continue
            formats_raw = item.get("formats")
            formats = [str(v) for v in formats_raw] if isinstance(formats_raw, list) else []
            out.append(make_profile(width, height, fps, sensor, formats, "frontend_cached"))
        return out

    color_clean = clean(color, "color")
    depth_clean = clean(depth, "depth")
    if not color_clean or not depth_clean:
        return None
    return {
        "source": "frontend_cached",
        "profile_url": profiles.get("profile_url"),
        "color": color_clean,
        "depth": depth_clean,
        "warning": "使用设置页已枚举的 SDK profile 校验，保存时不重复访问 /stream/profiles。",
    }


def apply_orbbec_settings(payload: dict[str, Any]) -> dict[str, Any]:
    apply_started = time.monotonic()
    timings: dict[str, float] = {}

    def mark(name: str, started: float) -> None:
        timings[name] = round((time.monotonic() - started) * 1000, 3)

    if str(payload.get("camera_model", "orbbec336l")) not in {"orbbec336l", "auto", ""}:
        raise ValueError("当前设置 API 只支持 Orbbec Gemini 336L")
    rgb = parse_orbbec_profile(str(payload.get("rgb_profile") or ""))
    depth = parse_orbbec_profile(str(payload.get("depth_profile") or ""))
    if rgb["fps"] != depth["fps"]:
        raise ValueError("当前 Orbbec Bridge env 只有一个 VISIONOPS_ORBBEC336L_FPS，RGB 与 Depth FPS 必须一致")

    path = orbbec_env_path()
    step_started = time.monotonic()
    values, lines = read_env_file(path)
    mark("read_env_ms", step_started)

    step_started = time.monotonic()
    profiles = profiles_from_payload(payload)
    if profiles is None:
        # Fallback only. Normal Web flow sends known_profiles from the previous GET request.
        profiles = collect_orbbec_profiles(values, timeout_s=1.0)
    mark("profile_validation_ms", step_started)

    if profiles.get("source") in {"bridge_api", "frontend_cached"}:
        if not validate_profile_against(rgb, profiles.get("color", [])):
            raise ValueError(f"RGB profile 不在 SDK 支持列表中: {profile_id(**rgb)}")
        if not validate_profile_against(depth, profiles.get("depth", [])):
            raise ValueError(f"Depth profile 不在 SDK 支持列表中: {profile_id(**depth)}")

    jpeg_quality = clamp_int(payload.get("camera_jpeg_quality"), 85, 10, 100)
    display_fps = clamp_int(payload.get("display_fps"), clamp_int(values.get("VISIONOPS_ORBBEC336L_MJPEG_FPS"), 10, 1, 30), 1, 30)
    depth_unit = str(payload.get("depth_unit") or values.get("VISIONOPS_ORBBEC336L_DEPTH_UNIT") or "mm").strip()
    if depth_unit not in {"mm", "raw_uint16"}:
        raise ValueError("depth_unit 只支持 mm 或 raw_uint16")

    updates = {
        "VISIONOPS_ORBBEC336L_COLOR_WIDTH": str(rgb["width"]),
        "VISIONOPS_ORBBEC336L_COLOR_HEIGHT": str(rgb["height"]),
        "VISIONOPS_ORBBEC336L_DEPTH_WIDTH": str(depth["width"]),
        "VISIONOPS_ORBBEC336L_DEPTH_HEIGHT": str(depth["height"]),
        "VISIONOPS_ORBBEC336L_FPS": str(rgb["fps"]),
        "VISIONOPS_ORBBEC336L_JPEG_QUALITY": str(jpeg_quality),
        "VISIONOPS_ORBBEC336L_MJPEG_FPS": str(display_fps),
        "VISIONOPS_ORBBEC336L_FLIP_VERTICAL": "true" if parse_bool(payload.get("flip_vertical"), False) else "false",
        "VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL": "true" if parse_bool(payload.get("flip_horizontal"), False) else "false",
        "VISIONOPS_ORBBEC336L_DEPTH_UNIT": depth_unit,
        "VISIONOPS_ORBBEC336L_SERIAL": str(payload.get("orbbec_serial") or "").strip(),
    }

    changed_keys = changed_env_keys(values, updates)
    merged_values = {**values, **updates}

    if not changed_keys:
        timings["write_env_ms"] = 0.0
        timings["restart_service_ms"] = 0.0
        timings["wait_health_ms"] = 0.0
        step_started = time.monotonic()
        service = systemd_status()
        mark("systemd_status_ms", step_started)
        timings["total_apply_ms"] = round((time.monotonic() - apply_started) * 1000, 3)
        return {
            "schema_version": "1.0",
            "message_type": "sdk_bridge_settings_apply_result",
            "status": "ok",
            "camera_model": "orbbec336l",
            "env_path": str(path),
            "backup_path": None,
            "backup_enabled": False,
            "changed": False,
            "changed_keys": [],
            "skipped_restart": True,
            "applied": updates,
            "restart": {"ok": True, "skipped": True, "reason": "env unchanged"},
            "health": {"ok": True, "skipped": True, "reason": "env unchanged"},
            "service": service,
            "settings": current_settings(merged_values),
            "profiles": profiles,
            "profile_refresh_skipped_after_apply": True,
            "apply_timings_ms": timings,
        }

    step_started = time.monotonic()
    write_env_file(path, lines, updates)
    mark("write_env_ms", step_started)

    step_started = time.monotonic()
    restart = restart_orbbec_service()
    mark("restart_service_ms", step_started)

    step_started = time.monotonic()
    health = wait_bridge_health(merged_values, timeout_s=4.0) if restart.get("ok") else {"ok": False, "error": "service restart failed"}
    mark("wait_health_ms", step_started)

    step_started = time.monotonic()
    service = systemd_status()
    mark("systemd_status_ms", step_started)
    timings["total_apply_ms"] = round((time.monotonic() - apply_started) * 1000, 3)

    # Do not call /stream/profiles again here. SDK profile enumeration can be slow and was already done on GET.
    return {
        "schema_version": "1.0",
        "message_type": "sdk_bridge_settings_apply_result",
        "status": "ok" if restart.get("ok") and health.get("ok") else "error",
        "camera_model": "orbbec336l",
        "env_path": str(path),
        "backup_path": None,
        "backup_enabled": False,
        "changed": True,
        "changed_keys": changed_keys,
        "skipped_restart": False,
        "applied": updates,
        "restart": restart,
        "health": health,
        "service": service,
        "settings": current_settings(merged_values),
        "profiles": profiles,
        "profile_refresh_skipped_after_apply": True,
        "apply_timings_ms": timings,
    }

