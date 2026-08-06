from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from production.common.runtime_ipc import TimedHttpResponse
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision import online_validate
from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.rgbd_cache import RgbdFrame


TIMESTAMP = 1785755358752


def _runtime_result():
    return {
        "schema_version": "1.0",
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "segmentation",
        "capture_timestamp_ms": TIMESTAMP,
        "result_id": "result-test",
        "frame_id": "frame-test",
        "image": {"width": 64, "height": 48},
        "input_roi": {
            "enabled": True,
            "pixel_xyxy": [8, 4, 40, 36],
            "crop_resolution": {"width": 32, "height": 32},
        },
        "model": {"model_id": "ring-test", "task_type": "segmentation"},
        "timing": {"inference_ms": 40.0, "postprocess_ms": 12.0, "total_ms": 56.0},
        "detections": [
            {
                "id": "seg-rknn-001",
                "class_id": 0,
                "class_name": "foam_ring",
                "score": 0.95,
                "mask": {
                    "source": "proto",
                    "size": [48, 64],
                    "polygon": [[[10, 8], [30, 8], [30, 28], [10, 28]]],
                },
            }
        ],
    }


class _FakeClient:
    def __init__(self, base_url, timeout_s, settings):
        self.base_url = base_url

    def status(self):
        return {
            "running": False,
            "mode": "idle",
            "health": "ok",
            "loaded_model": {"task_type": "segmentation", "model_id": "ring-test"},
            "frame_source": {
                "configured_transport": "posix_shared_memory",
                "transport": "posix_shared_memory",
                "fallback_active": False,
            },
        }

    def infer_once_raw(self):
        body = json.dumps(_runtime_result()).encode("utf-8")
        return TimedHttpResponse(
            body=body,
            status_code=200,
            headers={"content-length": str(len(body))},
            connect_ms=0.1,
            send_ms=0.1,
            headers_wait_ms=1.0,
            body_read_ms=0.2,
            total_ms=1.4,
            transport="raw_socket",
        )

    @staticmethod
    def decode_inference(raw):
        return json.loads(raw.decode("utf-8"))

    def transport_status(self):
        return {"last_transport": "raw_socket"}


class _FakeCache:
    def __init__(self, **_kwargs):
        self.frame = RgbdFrame(
            timestamp_epoch_ms=TIMESTAMP,
            rgb_sequence=101,
            depth_sequence=99,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            depth_mm=np.full((48, 64), 540, dtype=np.uint16),
            width=64,
            height=48,
            fx=600.0,
            fy=601.0,
            cx=32.0,
            cy=24.0,
            aligned_to_color=True,
            calibration_ready=True,
            flip_horizontal=False,
            flip_vertical=False,
            cached_monotonic=time.monotonic(),
        )

    def start(self):
        return None

    def stop(self):
        return None

    def wait_until_ready(self, _timeout):
        return True

    def get_exact(self, timestamp, timeout=0.0):
        assert timestamp == TIMESTAMP
        return self.frame

    def status(self):
        return {"running": True, "cache_size": 1, "exact_match_count": 1}


def test_online_once_uses_exact_frame_and_writes_result(tmp_path, monkeypatch):
    config = tmp_path / "line.yaml"
    config.write_text(
        """
classes:
  foam_ring: foam_ring
  ring_mouth: ring_mouth
box_wall:
  enabled: false
online_rgbd:
  enabled: true
  cache_frames: 4
  cache_rgb: true
  require_exact_timestamp: true
  allow_nearest_fallback: false
online_geometry:
  runtime_url: http://127.0.0.1:28081
  save_exact_rgb_png: false
  save_exact_depth_png: false
  save_depth_colormap: false
  save_runtime_result: false
  save_overlay: false
axis_direction:
  enabled: false
full_gripper_motion_collision:
  enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(online_validate, "RuntimeIpcClient", _FakeClient)
    monkeypatch.setattr(online_validate, "RgbdFrameCache", _FakeCache)
    monkeypatch.setattr(
        online_validate,
        "analyze_scene",
        lambda instances, depth, intrinsics, config: {
            "rings_detected": len(instances),
            "mouths_detected": 0,
            "matched_pairs": 0,
            "eligible_count": 0,
            "selected_ring_instance_id": None,
            "selected_clock_hour": None,
            "robot_candidate": None,
            "instances": [],
        },
    )

    payload = online_validate.run_once(
        config_path=config,
        output_root=tmp_path / "out",
    )
    assert payload["status"] == "ok"
    assert payload["rgbd_match"]["timestamp_delta_ms"] == 0
    assert payload["segmentation_adaptation"]["accepted_count"] == 1
    assert payload["coordinate_space"]["geometry_roi_xyxy"] == [8, 4, 40, 36]
    assert payload["image"] == {"width": 32, "height": 32}
    assert payload["intrinsics"]["cx"] == 24.0
    assert payload["intrinsics"]["cy"] == 20.0
    assert payload["robot_ready"] is False
    result = tmp_path / "out" / str(TIMESTAMP) / "online_geometry_result.json"
    assert result.exists()
    saved = json.loads(result.read_text(encoding="utf-8"))
    assert saved["capture_timestamp_ms"] == TIMESTAMP


def test_online_validate_direct_script_can_import_repository():
    script = Path(online_validate.__file__).resolve()
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
    assert "M38.6" in completed.stdout
