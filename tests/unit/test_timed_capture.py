"""Collector 定时采图控制器测试。"""

from __future__ import annotations

import time

from apps.collector_web.backend import timed_capture as module


class _FakeRuntimeClient:
    pass


def test_timed_capture_saves_repeated_snapshots_and_stops(monkeypatch) -> None:
    calls: list[float] = []

    def fake_save(_client, prefix: str = "visionops"):
        calls.append(time.monotonic())
        return {"image": {"filename": f"{prefix}_{len(calls)}.jpg"}}

    monkeypatch.setattr(module, "save_runtime_snapshot", fake_save)
    controller = module.TimedCaptureController(_FakeRuntimeClient())
    try:
        started = controller.start(0.5)
        assert started["enabled"] is True
        deadline = time.monotonic() + 2.0
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(calls) >= 2
        status = controller.status()
        assert status["capture_count"] >= 2
        assert status["last_image"]["filename"].startswith("visionops_auto_")

        controller.stop()
        stopped_at = len(calls)
        time.sleep(0.65)
        assert len(calls) == stopped_at
        assert controller.status()["enabled"] is False
    finally:
        controller.close()
