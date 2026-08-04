from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.online_processor import (
    OnlineProcessResult,
)
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision import service as service_module


class _Prepared:
    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id
        self.prepared_monotonic = time.monotonic()


class _FakeProcessor:
    def __init__(self, delay_s: float = 0.01) -> None:
        self.delay_s = delay_s
        self.started = False
        self.calls: list[tuple[str | None, bool | None]] = []
        self.raw_config = {"geometry_optimization": {"mode": "first_valid"}}
        self.runtime_url = "http://127.0.0.1:28081"
        self._release = threading.Event()
        self.block = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False
        self._release.set()

    def prepare(self, *, request_id):
        self.calls.append((request_id, None))
        return _Prepared(request_id)

    def finish(
        self,
        prepared,
        *,
        save_debug,
        generate_overlay,
        stage,
        geometry_queue_wait_ms=None,
    ):
        request_id = prepared.request_id
        self.calls[-1] = (request_id, save_debug)
        if self.block:
            self._release.wait(timeout=2.0)
        else:
            time.sleep(self.delay_s)
        files = {"geometry_result": "/tmp/debug.json"} if save_debug else {}
        payload = {
            "status": "ok",
            "request_id": request_id,
            "robot_ready_reason": "test",
            "capture_timestamp_ms": 1000 + len(self.calls),
            "rgbd_match": {"timestamp_delta_ms": 0},
            "runtime": {
                "result_id": f"result-{len(self.calls)}",
                "frame_id": f"frame-{len(self.calls)}",
                "timing": {"total_ms": 50.0},
            },
            "timing_ms": {
                "prepare_total_ms": 20.0,
                "polygon_to_mask_ms": 10.0,
                "geometry_ms": 100.0,
                "visualization_ms": 5.0,
                "save_outputs_ms": 0.0,
                "total_ms": 170.0,
            },
            "scene": {
                "rings_detected": 2,
                "mouths_detected": 1,
                "matched_pairs": 1,
                "eligible_count": 1,
                "selected_ring_instance_id": 3,
                "selected_clock_hour": 5,
                "selected_clock_angle_deg_cw_from_12": 150.0,
                "selected_clock_search_batch": "primary",
                "geometry_optimization": {
                    "mode": "first_valid",
                    "fully_analyzed_pair_count": 1,
                    "full_candidate_evaluated_count": 1,
                    "adaptive_fallback_used": False,
                    "early_exit_triggered": True,
                },
            },
            "candidate": {"target": {"ring_instance_id": 3}},
            "files": files,
        }
        return OnlineProcessResult(
            payload=payload,
            overlay_jpeg=b"\xff\xd8" + b"x" * 200 + b"\xff\xd9" if generate_overlay else None,
        )

    def process(self, *, request_id, save_debug, generate_overlay, stage):
        prepared = self.prepare(request_id=request_id)
        return self.finish(
            prepared,
            save_debug=save_debug,
            generate_overlay=generate_overlay,
            stage=stage,
            geometry_queue_wait_ms=0.0,
        )

    def status(self, *, refresh_runtime=False):
        return {
            "started": self.started,
            "runtime": {
                "loaded_model": {"task_type": "segmentation"},
                "frame_source": {
                    "transport": "posix_shared_memory",
                    "fallback_active": False,
                },
            },
            "runtime_error": None,
            "runtime_ipc": {"last_transport": "raw_socket"},
            "cache": {"running": self.started, "pair_fps": 30.0, "last_error": None},
        }


def _settings(queue_capacity: int = 4):
    return {
        "trigger_queue_capacity": queue_capacity,
        "geometry_queue_capacity": queue_capacity,
        "request_registry_capacity": 16,
        "default_wait_timeout_ms": 5000,
        "maximum_wait_timeout_ms": 10000,
        "default_save_debug": False,
        "latest_overlay_enabled": True,
        "overlay_jpeg_quality": 90,
        "runtime_status_ttl_ms": 1000,
        "output_root": "/tmp",
        "max_http_body_bytes": 1024 * 1024,
    }


def test_persistent_service_replays_duplicate_without_reprocessing():
    processor = _FakeProcessor()
    service = service_module.FoamRingOnlineService(
        processor=processor,
        settings=_settings(),
    )
    service.start()
    try:
        first, replay = service.submit(request_id="robot-1", save_debug=False)
        assert replay is False
        assert first.done.wait(2.0)
        assert first.status == "ok"
        second, replay = service.submit(request_id="robot-1", save_debug=True)
        assert replay is True
        assert second is first
        assert len(processor.calls) == 1
        document = second.status_document(replay=True)
        assert document["idempotent_replay"] is True
        assert document["capture_timestamp_ms"] == first.result["capture_timestamp_ms"]
    finally:
        service.stop()


def test_production_trigger_does_not_write_debug_and_snapshot_is_retained():
    processor = _FakeProcessor()
    service = service_module.FoamRingOnlineService(
        processor=processor,
        settings=_settings(),
    )
    service.start()
    try:
        job, _ = service.submit(request_id="robot-2", save_debug=None)
        assert job.done.wait(2.0)
        assert job.result is not None
        assert job.result["files"] == {}
        assert job.result["target_found"] is True
        jpeg, timestamp = service.snapshot()
        assert jpeg is not None and jpeg.startswith(b"\xff\xd8")
        assert timestamp == job.result["capture_timestamp_ms"]
        health = service.health_document()
        assert health["health"] == "ok"
    finally:
        service.stop()


def test_queue_capacity_rejects_new_explicit_job_without_overwriting_existing():
    processor = _FakeProcessor()
    processor.block = True
    service = service_module.FoamRingOnlineService(
        processor=processor,
        settings=_settings(queue_capacity=1),
    )
    service.start()
    try:
        running, _ = service.submit(request_id="running", save_debug=False)
        deadline = time.time() + 1.0
        while running.status != "running" and time.time() < deadline:
            time.sleep(0.01)
        queued, _ = service.submit(request_id="queued", save_debug=False)
        try:
            service.submit(request_id="overflow", save_debug=False)
        except service_module.QueueCapacityError:
            pass
        else:
            raise AssertionError("queue overflow must be rejected")
        assert service.get_request("running") is running
        assert service.get_request("queued") is queued
    finally:
        processor._release.set()
        service.stop()


def test_service_direct_script_can_import_repository():
    script = Path(service_module.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "M36.5" in completed.stdout
