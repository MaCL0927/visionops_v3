"""Collector generic-runtime FPS scheduling regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from apps.collector_web.backend.config_loader import CollectorConfig
from apps.collector_web.backend import vision_box_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_realtime_loops_do_not_wait_for_snapshot_refresh() -> None:
    production = (
        PROJECT_ROOT
        / "apps/collector_web/frontend/static/js/pages/production.js"
    ).read_text(encoding="utf-8")
    validate = (
        PROJECT_ROOT
        / "apps/collector_web/frontend/static/js/pages/validate.js"
    ).read_text(encoding="utf-8")

    assert "productionInferOnce({ refreshSnapshot: false })" in production
    assert "scheduleSnapshotLoop(0)" in production
    assert "await displaySnapshot();" not in production.split(
        "async function runLiveLoop()", 1
    )[1].split("function scheduleSnapshotLoop", 1)[0]

    assert 'inferOnce({ refreshSnapshot: false, measureRate: true })' in validate
    assert "scheduleRealtimeSnapshot(0)" in validate
    assert "class SlidingRateMeter" in (
        PROJECT_ROOT / "apps/collector_web/frontend/static/js/realtime_rate.js"
    ).read_text(encoding="utf-8")


def test_production_inference_fps_is_persisted_for_generic_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "vision_box_settings.json"
    monkeypatch.setattr(vision_box_settings, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        vision_box_settings,
        "_normalize_network",
        lambda raw: {
            "eth0": {"interface": "eth0", "ip": "", "netmask": "", "gateway": ""},
            "eth1": {"interface": "eth1", "ip": "", "netmask": "", "gateway": ""},
        },
    )
    monkeypatch.setattr(
        vision_box_settings,
        "_network_differs_from_live",
        lambda _network: False,
    )
    monkeypatch.setattr(
        vision_box_settings,
        "read_dual_nic_state",
        lambda: {"mode": "test", "items": [], "interfaces": {}},
    )

    config = CollectorConfig(
        host="127.0.0.1",
        port=18091,
        runtime_url="http://127.0.0.1:28081",
        gateway_url="http://127.0.0.1:19090",
        business_app_url="http://127.0.0.1:19110",
        device_id="test-device",
        models_root=str(tmp_path / "models"),
        production_inference_source="runtime",
    )

    result = vision_box_settings.apply_vision_box_settings(
        config,
        {
            "default_mode": "factory",
            "production_inference_fps": 30,
            "status_refresh_fps": 0.5,
            "disk_warning_percent": 85,
        },
    )

    assert result["settings"]["production_inference_fps"] == 30.0
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["production_inference_fps"] == 30.0
    assert vision_box_settings.load_vision_box_settings(config)[
        "production_inference_fps"
    ] == 30.0
