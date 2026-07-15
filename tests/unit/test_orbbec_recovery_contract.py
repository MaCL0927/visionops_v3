"""Static contracts for the 7x24 Orbbec camera recovery path."""
from __future__ import annotations

from pathlib import Path

from production.carton_line.gateway.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bridge_source_contains_freshness_reconnect_state_machine() -> None:
    source = (
        PROJECT_ROOT
        / "edge/camera_bridge/orbbec336l_bridge/visionops_orbbec336l_bridge.cpp"
    ).read_text(encoding="utf-8")
    for token in (
        "VISIONOPS_ORBBEC336L_STALE_TIMEOUT_MS",
        "CameraState::Reconnecting",
        "camera_connected",
        "CAMERA_FRAME_STALE",
        "reconnect_attempt_count",
        "invalidate_frames_locked",
        "camera frame stale or reconnecting",
    ):
        assert token in source


def test_bridge_watchdog_units_and_script_are_installed_by_profile() -> None:
    installer = (
        PROJECT_ROOT / "production/carton_line/deploy/install_services.sh"
    ).read_text(encoding="utf-8")
    script = PROJECT_ROOT / "production/carton_line/scripts/watch_orbbec336l_bridge.sh"
    service = (
        PROJECT_ROOT
        / "production/carton_line/deploy/systemd/visionops-orbbec336l-bridge-watchdog.service"
    )
    timer = (
        PROJECT_ROOT
        / "production/carton_line/deploy/systemd/visionops-orbbec336l-bridge-watchdog.timer"
    )
    assert script.exists()
    assert service.exists()
    assert timer.exists()
    assert "visionops-orbbec336l-bridge-watchdog.timer" in installer
    assert "OnUnitActiveSec=10s" in timer.read_text(encoding="utf-8")


def test_camera_alarm_modbus_interface_is_reserved_but_disabled() -> None:
    config = load_config(
        str(PROJECT_ROOT / "production/carton_line/config/line.yaml")
    )
    camera_health = config["pick"]["camera_health"]
    assert camera_health["suppress_detection_when_unhealthy"] is True
    assert camera_health["modbus_tcp"]["reserved"] is True
    assert camera_health["modbus_tcp"]["enabled"] is False
