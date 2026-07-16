from __future__ import annotations

import json
from pathlib import Path

from edge.camera_bridge import camera_selection
from apps.collector_web.backend import sdk_bridge_settings


def test_active_camera_switch_updates_all_bridge_urls(tmp_path: Path, monkeypatch) -> None:
    selection = tmp_path / "active_camera.json"
    monkeypatch.setenv("VISIONOPS_CAMERA_SELECTION_FILE", str(selection))
    camera_selection.write_camera_selection("hp60c")
    config = {
        "camera_bridge": {
            "base_url": "http://127.0.0.1:18182",
            "depth_url": "http://127.0.0.1:18182/stream/depth.png",
            "depth_meta_url": "http://127.0.0.1:18182/stream/depth_meta",
            "deproject_url": "http://127.0.0.1:18182/api/coordinate/deproject",
        },
        "pick": {"video": {"public_url": "http://192.168.213.137:18182/stream.mjpeg"}},
    }
    camera_selection.apply_active_camera_to_config(config)
    assert config["camera_bridge"]["camera_model"] == "hp60c"
    assert config["camera_bridge"]["base_url"] == "http://127.0.0.1:18181"
    assert config["camera_bridge"]["depth_url"].endswith(":18181/stream/depth.png")
    assert config["pick"]["video"]["public_url"] == "http://192.168.213.137:18181/stream.mjpeg"


def test_apply_hp60c_settings_switches_selection_and_writes_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / "hp60c.env"
    env_path.write_text(
        "VISIONOPS_HP60C_HTTP_HOST=0.0.0.0\n"
        "VISIONOPS_HP60C_HTTP_PORT=18181\n"
        "VISIONOPS_HP60C_COLOR_WIDTH=640\n"
        "VISIONOPS_HP60C_COLOR_HEIGHT=480\n"
        "VISIONOPS_HP60C_DEPTH_WIDTH=640\n"
        "VISIONOPS_HP60C_DEPTH_HEIGHT=480\n"
        "VISIONOPS_HP60C_FPS=30\n",
        encoding="utf-8",
    )
    selection = tmp_path / "active_camera.json"
    monkeypatch.setenv("VISIONOPS_CAMERA_SELECTION_FILE", str(selection))
    sdk_bridge_settings.CAMERA_CONFIG["hp60c"]["env_path"] = env_path
    monkeypatch.setattr(sdk_bridge_settings, "restart_service", lambda service: {"ok": True, "service": service})
    monkeypatch.setattr(sdk_bridge_settings, "wait_bridge_health", lambda model, values: {"ok": True})
    monkeypatch.setattr(sdk_bridge_settings, "restart_camera_consumers", lambda: [
        {"ok": True, "service": "runtime", "role": "runtime"},
        {"ok": True, "service": "business", "role": "business"},
    ])
    monkeypatch.setattr(sdk_bridge_settings, "collect_profiles", lambda model, values: {
        "source": "env_current_fallback", "profile_url": "", "warning": None,
        "color": [sdk_bridge_settings.make_profile(model, 640, 480, 30, "color")],
        "depth": [sdk_bridge_settings.make_profile(model, 640, 480, 30, "depth", ["Y16"])],
    })
    monkeypatch.setattr(sdk_bridge_settings, "service_status", lambda service: {"name": service, "active": "active", "enabled": "enabled", "active_ok": True})

    result = sdk_bridge_settings.apply_sdk_bridge_settings({
        "camera_model": "hp60c",
        "rgb_profile": "hp60c:640x480@30",
        "depth_profile": "hp60c:640x480@30",
        "display_fps": 10,
        "camera_jpeg_quality": 85,
        "flip_vertical": True,
        "flip_horizontal": False,
        "depth_unit": "mm",
        "rgb_source_preference": "mjpeg",
        "rgb_order": "bgr",
        "hp60c_config_path": "/tmp/hp60c.json",
        "hp60c_fx": 500.1,
        "hp60c_fy": 501.2,
        "hp60c_cx": 319.5,
        "hp60c_cy": 239.5,
    })
    assert result["active_camera"] == "hp60c"
    assert json.loads(selection.read_text(encoding="utf-8"))["active_camera"] == "hp60c"
    text = env_path.read_text(encoding="utf-8")
    assert "VISIONOPS_HP60C_RGB_SOURCE=mjpeg" in text
    assert "VISIONOPS_HP60C_CONFIG=/tmp/hp60c.json" in text
    assert "VISIONOPS_HP60C_FX=500.1" in text
    assert "VISIONOPS_HP60C_CY=239.5" in text


def test_hp60c_bridge_contract_contains_depth_and_recovery() -> None:
    source = (Path(__file__).resolve().parents[2] / "edge/camera_bridge/hp60c_bridge/visionops_hp60c_sdk_bridge.cpp").read_text(encoding="utf-8")
    for token in (
        "/stream/depth.png", "/stream/depth_meta", "/stream/depth_vis.jpg",
        "recovery_loop", "reconnect_attempt_count", "camera frame stale",
        "VISIONOPS_HP60C_STALE_TIMEOUT_MS",
    ):
        assert token in source


def test_failed_bridge_start_does_not_publish_camera_switch(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / "hp60c.env"
    env_path.write_text(
        "VISIONOPS_HP60C_HTTP_HOST=0.0.0.0\n"
        "VISIONOPS_HP60C_HTTP_PORT=18181\n"
        "VISIONOPS_HP60C_COLOR_WIDTH=640\n"
        "VISIONOPS_HP60C_COLOR_HEIGHT=480\n"
        "VISIONOPS_HP60C_DEPTH_WIDTH=640\n"
        "VISIONOPS_HP60C_DEPTH_HEIGHT=480\n"
        "VISIONOPS_HP60C_FPS=30\n",
        encoding="utf-8",
    )
    selection = tmp_path / "active_camera.json"
    monkeypatch.setenv("VISIONOPS_CAMERA_SELECTION_FILE", str(selection))
    camera_selection.write_camera_selection("orbbec336l")
    sdk_bridge_settings.CAMERA_CONFIG["hp60c"]["env_path"] = env_path
    monkeypatch.setattr(sdk_bridge_settings, "restart_service", lambda service: {"ok": False, "stderr": "not installed"})

    import pytest
    with pytest.raises(RuntimeError, match="重启"):
        sdk_bridge_settings.apply_sdk_bridge_settings({
            "camera_model": "hp60c",
            "rgb_profile": "hp60c:640x480@30",
            "depth_profile": "hp60c:640x480@30",
        })
    assert json.loads(selection.read_text(encoding="utf-8"))["active_camera"] == "orbbec336l"


def test_hp60c_watchdog_ignores_uninstalled_bridge() -> None:
    script = (Path(__file__).resolve().parents[2] / "production/carton_line/scripts/watch_hp60c_bridge.sh").read_text(encoding="utf-8")
    assert 'systemctl cat "$BRIDGE_SERVICE"' in script
    assert "must never advance reboot counters" in script
