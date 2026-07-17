"""Shared active-camera selection for VisionOps edge services.

Both SDK bridges may stay online at the same time. The selected camera only decides
which bridge URL Runtime/business services consume. Collector Web writes this file,
and launchers read it on every service start.
"""

from __future__ import annotations

import json
import os
import tempfile
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION_PATH = PROJECT_ROOT / "config" / "active_camera.json"

CAMERA_SPECS: dict[str, dict[str, Any]] = {
    "orbbec336l": {
        "camera_model": "orbbec336l",
        "display_name": "Orbbec Gemini 336L",
        "base_url": "http://127.0.0.1:18182",
        "public_port": 18182,
        "service": "visionops-orbbec336l-bridge.service",
        "watchdog_timer": "visionops-orbbec336l-bridge-watchdog.timer",
        "snapshot_path": "/stream/snapshot.jpg",
        "depth_path": "/stream/depth.png",
        "depth_meta_path": "/stream/depth_meta",
        "health_path": "/health",
        "profiles_path": "/stream/profiles",
        "mjpeg_path": "/stream.mjpeg",
        "camera_info_path": "/stream/camera_info",
        "deproject_path": "/api/coordinate/deproject",
    },
    "hp60c": {
        "camera_model": "hp60c",
        "display_name": "HP60C / HP60CN",
        "base_url": "http://127.0.0.1:18181",
        "public_port": 18181,
        "service": "visionops-hp60c-sdk-bridge.service",
        "watchdog_timer": "visionops-hp60c-sdk-bridge-watchdog.timer",
        "snapshot_path": "/stream/snapshot.jpg",
        "depth_path": "/stream/depth.png",
        "depth_meta_path": "/stream/depth_meta",
        "health_path": "/health",
        "profiles_path": "/stream/profiles",
        "mjpeg_path": "/stream.mjpeg",
        "camera_info_path": "/stream/camera_info",
        "deproject_path": "/api/coordinate/deproject",
    },
}


def selection_path() -> Path:
    return Path(os.environ.get("VISIONOPS_CAMERA_SELECTION_FILE", str(DEFAULT_SELECTION_PATH))).expanduser()


def normalize_camera_model(value: object, fallback: str = "orbbec336l") -> str:
    model = str(value or "").strip().lower()
    aliases = {
        "orbbec": "orbbec336l",
        "336l": "orbbec336l",
        "gemini336l": "orbbec336l",
        "hp60cn": "hp60c",
        "auto": fallback,
        "": fallback,
    }
    model = aliases.get(model, model)
    return model if model in CAMERA_SPECS else fallback


def _default_document() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "active_camera": "orbbec336l",
        "cameras": deepcopy(CAMERA_SPECS),
    }


def read_camera_selection(path: Path | None = None) -> dict[str, Any]:
    target = path or selection_path()
    document = _default_document()
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}
    if isinstance(loaded, Mapping):
        active = normalize_camera_model(loaded.get("active_camera"), "orbbec336l")
        document["active_camera"] = active
        cameras = loaded.get("cameras")
        if isinstance(cameras, Mapping):
            for model, overrides in cameras.items():
                normalized = normalize_camera_model(model, "")
                if normalized in CAMERA_SPECS and isinstance(overrides, Mapping):
                    document["cameras"][normalized].update({
                        key: deepcopy(value)
                        for key, value in overrides.items()
                        if key in document["cameras"][normalized]
                    })
    document["path"] = str(target)
    return document


def write_camera_selection(camera_model: str, path: Path | None = None) -> dict[str, Any]:
    target = path or selection_path()
    model = normalize_camera_model(camera_model, "")
    if model not in CAMERA_SPECS:
        raise ValueError(f"不支持的相机型号: {camera_model}")
    document = read_camera_selection(target)
    changed = document.get("active_camera") != model
    document.pop("path", None)
    document["active_camera"] = model
    body = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        try:
            os.replace(tmp_name, target)
            os.chmod(target, 0o664)
            tmp_name = ""
        except PermissionError:
            command = ["install", "-m", "0664", tmp_name, str(target)] if os.geteuid() == 0 else ["sudo", "-n", "install", "-m", "0664", tmp_name, str(target)]
            proc = subprocess.run(command, text=True, capture_output=True, timeout=8, check=False)
            if proc.returncode != 0:
                raise PermissionError(proc.stderr.strip() or f"无法写入 {target}")
    finally:
        try:
            if tmp_name and os.path.exists(tmp_name): os.unlink(tmp_name)
        except OSError:
            pass
    result = read_camera_selection(target)
    result["changed"] = changed
    return result


def active_camera_spec(path: Path | None = None) -> dict[str, Any]:
    document = read_camera_selection(path)
    model = normalize_camera_model(document.get("active_camera"), "orbbec336l")
    spec = deepcopy(document["cameras"].get(model, CAMERA_SPECS[model]))
    spec["camera_model"] = model
    spec["selection_path"] = document["path"]
    return spec


def public_mjpeg_url(existing_url: object, spec: Mapping[str, Any]) -> str:
    """Preserve configured public host while switching the bridge port/path."""
    value = str(existing_url or "").strip()
    parsed = urlparse(value)
    host = parsed.hostname or "127.0.0.1"
    scheme = parsed.scheme or "http"
    port = int(spec.get("public_port") or urlparse(str(spec["base_url"])).port or 80)
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    netloc = f"{netloc}:{port}"
    return urlunparse((scheme, netloc, str(spec.get("mjpeg_path") or "/stream.mjpeg"), "", "", ""))


def apply_active_camera_to_config(config: dict[str, Any]) -> dict[str, Any]:
    """Mutate a merged production config so all downstream consumers use one bridge."""
    spec = active_camera_spec()
    bridge = config.get("camera_bridge")
    if isinstance(bridge, dict):
        bridge["camera_model"] = spec["camera_model"]
        bridge["base_url"] = spec["base_url"]
        for key in ("snapshot_path", "depth_path", "depth_meta_path", "health_path", "camera_info_path", "deproject_path"):
            if key in spec:
                bridge[key] = spec[key]
        bridge["depth_url"] = str(spec["base_url"]).rstrip("/") + str(spec["depth_path"])
        bridge["depth_meta_url"] = str(spec["base_url"]).rstrip("/") + str(spec["depth_meta_path"])
        bridge["deproject_url"] = str(spec["base_url"]).rstrip("/") + str(spec["deproject_path"])
        bridge["camera_info_url"] = str(spec["base_url"]).rstrip("/") + str(spec["camera_info_path"])
        bridge["service"] = spec["service"]
    pick = config.get("pick")
    if isinstance(pick, dict):
        video = pick.get("video")
        if isinstance(video, dict):
            video["public_url"] = public_mjpeg_url(video.get("public_url"), spec)
    box_grasp = config.get("box_grasp")
    if isinstance(box_grasp, dict):
        video = box_grasp.get("video")
        if isinstance(video, dict):
            video["public_url"] = public_mjpeg_url(video.get("public_url"), spec)
    config["active_camera"] = {
        "camera_model": spec["camera_model"],
        "display_name": spec["display_name"],
        "base_url": spec["base_url"],
        "service": spec["service"],
        "selection_path": spec["selection_path"],
    }
    return config
