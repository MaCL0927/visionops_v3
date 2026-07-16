"""Behavioral tests for whole-box reboot escalation after camera recovery failures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = PROJECT_ROOT / "production/carton_line/scripts/watch_orbbec336l_bridge.sh"


def _make_fake_commands(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    actions = tmp_path / "systemctl-actions.log"

    (fake_bin / "systemctl").write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "${FAKE_SYSTEMCTL_ACTIONS}"
if [[ "${1:-}" == "is-active" ]]; then
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    (fake_bin / "curl").write_text(
        """#!/usr/bin/env bash
set -eu
if [[ -n "${FAKE_CURL_BODY:-}" ]]; then
  printf '%s' "${FAKE_CURL_BODY}"
  exit 0
fi
exit 22
""",
        encoding="utf-8",
    )
    (fake_bin / "logger").write_text(
        """#!/usr/bin/env bash
exit 0
""",
        encoding="utf-8",
    )
    for path in fake_bin.iterdir():
        path.chmod(0o755)
    return fake_bin, actions


def _base_env(tmp_path: Path, fake_bin: Path, actions: Path) -> dict[str, str]:
    run_dir = tmp_path / "run"
    persist_dir = tmp_path / "persist"
    run_dir.mkdir(exist_ok=True)
    persist_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_SYSTEMCTL_ACTIONS": str(actions),
            "VISIONOPS_CAMERA_WATCHDOG_LOCK_FILE": str(run_dir / "watchdog.lock"),
            "VISIONOPS_CAMERA_WATCHDOG_STATE_FILE": str(run_dir / "watchdog.state"),
            "VISIONOPS_CAMERA_WATCHDOG_STAMP_FILE": str(run_dir / "last_restart"),
            "VISIONOPS_CAMERA_WATCHDOG_DISABLE_FILE": str(run_dir / "disabled"),
            "VISIONOPS_CAMERA_WATCHDOG_PERSIST_DIR": str(persist_dir),
            "VISIONOPS_CAMERA_WATCHDOG_COOLDOWN_S": "0",
            "VISIONOPS_CAMERA_WATCHDOG_RECOVERY_WAIT_S": "0",
            "VISIONOPS_CAMERA_WATCHDOG_RUNTIME_WAIT_S": "0",
            "VISIONOPS_CAMERA_WATCHDOG_UNHEALTHY_RESTART_AFTER_S": "0",
            "VISIONOPS_CAMERA_WATCHDOG_MAX_SERVICE_RESTARTS": "2",
            "VISIONOPS_CAMERA_WATCHDOG_REBOOT_ENABLED": "true",
            "VISIONOPS_CAMERA_WATCHDOG_REBOOT_DELAY_S": "0",
            "VISIONOPS_CAMERA_WATCHDOG_REBOOT_ONCE_PER_INCIDENT": "true",
        }
    )
    return env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WATCHDOG)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_reboots_once_after_configured_failed_bridge_service_restarts(tmp_path: Path) -> None:
    fake_bin, actions = _make_fake_commands(tmp_path)
    env = _base_env(tmp_path, fake_bin, actions)

    _run(env)

    action_lines = actions.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("restart ") for line in action_lines) == 1
    assert not any(line.startswith("reboot") for line in action_lines)

    _run(env)
    action_lines = actions.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("restart ") for line in action_lines) == 2
    assert sum(line.startswith("reboot") for line in action_lines) == 1

    # A still-unplugged camera after boot must not create a reboot loop for the
    # same persistent fault incident.
    _run(env)
    action_lines = actions.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("reboot") for line in action_lines) == 1


def test_healthy_camera_clears_restart_counter_and_reboot_marker(tmp_path: Path) -> None:
    fake_bin, actions = _make_fake_commands(tmp_path)
    env = _base_env(tmp_path, fake_bin, actions)
    persist_dir = Path(env["VISIONOPS_CAMERA_WATCHDOG_PERSIST_DIR"])

    for _ in range(2):
        _run(env)
    assert (persist_dir / "orbbec336l_reboot_issued").exists()

    env["FAKE_CURL_BODY"] = json.dumps(
        {
            "camera_connected": True,
            "camera_state": "running",
            "camera_thread_alive": True,
            "last_color_age_ms": 10,
            "last_depth_age_ms": 10,
            "reconnect_attempt_count": 0,
            "frame_count": 100,
        }
    )
    _run(env)

    assert (persist_dir / "orbbec336l_failed_service_restarts").read_text().strip() == "0"
    assert not (persist_dir / "orbbec336l_reboot_issued").exists()
    assert not (persist_dir / "orbbec336l_incident_started_at").exists()
