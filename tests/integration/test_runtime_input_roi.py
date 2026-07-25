"""M32.3 Runtime input ROI parsing, preprocessing and coordinate contract tests."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _request_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return json.loads(error.read().decode("utf-8"))


@contextmanager
def _running_runtime(binary: Path, extra_args: list[str]):
    port = _free_port()
    process = subprocess.Popen(
        [
            str(binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device-id",
            "input-roi-test",
            *extra_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Runtime 提前退出\nstdout={stdout}\nstderr={stderr}")
        try:
            _request_json(f"{base_url}/health")
            break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        process.wait(timeout=3)
        pytest.fail("Runtime 未在超时时间内启动")

    try:
        yield base_url
    finally:
        process.terminate()
        process.wait(timeout=3)
        assert process.returncode == 0


def _write_ppm(path: Path, width: int, height: int) -> None:
    # A deterministic RGB gradient makes the fixture suitable for later
    # pixel-level preprocessing tests without depending on OpenCV.
    row = bytearray()
    for x in range(width):
        row.extend((x % 251, (x * 3) % 251, (x * 7) % 251))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes(row) * height)


def _write_model_package(path: Path) -> None:
    path.mkdir()
    (path / "model.rknn").write_bytes(b"mock")
    (path / "model.yaml").write_text(
        """schema_version: '1.0'
model_id: rk3576-001_aaa_det_20260725_143034
model_name: detection-rk3576-001_aaa_det_20260725_142936
model_version: 20260725_143035
task_type: detection
target_platform: rk3576
input_size: [640, 640]
class_names:
- blue
- head
postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_det: 100
runtime:
  preprocess: letterbox
  color: rgb
preprocess:
  input_roi:
    enabled: true
    coordinate_space: runtime_snapshot
    source_resolution:
      width: 1280
      height: 720
    pixel_xyxy: [850, 490, 1277, 719]
    normalized_xyxy: [0.6640625, 0.680555555556, 0.99765625, 0.998611111111]
    crop_resolution:
      width: 427
      height: 229
    resize_mode: letterbox
    pad_value: 114
""",
        encoding="utf-8",
    )


def test_runtime_loads_input_roi_and_applies_it_to_exact_source_resolution(
    shared_runtime_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "model"
    image = tmp_path / "frame.ppm"
    _write_model_package(package)
    _write_ppm(image, 1280, 720)

    with _running_runtime(
        shared_runtime_binary,
        [
            "--model-dir",
            str(package),
            "--test-image",
            str(image),
            "--preprocess-backend",
            "cpu",
        ],
    ) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        configured = status["loaded_model"]["input_roi"]
        assert status["loaded_model"]["labels_count"] == 2
        assert configured["enabled"] is True
        assert configured["pixel_xyxy"] == [850, 490, 1277, 719]
        assert configured["crop_resolution"] == {"width": 427, "height": 229}
        assert status["preprocess"]["input_roi"] == configured

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "ok"
        assert result["image"] == {"width": 1280, "height": 720}
        assert result["input_roi"]["enabled"] is True
        assert result["input_roi"]["pixel_xyxy"] == [850, 490, 1277, 719]
        assert result["input_roi"]["scaled_from_normalized"] is False
        assert result["timing"]["input_roi_resolve_ms"] >= 0
        assert result["timing"]["crop_resize_ms"] >= 0
        assert result["timing_detail"]["crop_resize_ms"] >= 0


def test_runtime_scales_input_roi_from_normalized_coordinates_when_resolution_changes(
    shared_runtime_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "model"
    image = tmp_path / "frame.ppm"
    _write_model_package(package)
    _write_ppm(image, 640, 360)

    with _running_runtime(
        shared_runtime_binary,
        ["--model-dir", str(package), "--test-image", str(image)],
    ) as base_url:
        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "ok"
        assert result["image"] == {"width": 640, "height": 360}
        assert result["input_roi"]["scaled_from_normalized"] is True
        assert result["input_roi"]["pixel_xyxy"] == [425, 245, 639, 359]
        assert result["input_roi"]["crop_resolution"] == {"width": 214, "height": 114}


def test_runtime_rejects_input_roi_when_camera_aspect_ratio_changes(
    shared_runtime_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "model"
    image = tmp_path / "frame.ppm"
    _write_model_package(package)
    _write_ppm(image, 640, 480)

    with _running_runtime(
        shared_runtime_binary,
        ["--model-dir", str(package), "--test-image", str(image)],
    ) as base_url:
        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "error"
        assert result["error"]["code"] == "INPUT_ROI_SOURCE_ASPECT_MISMATCH"
        assert result["image"] == {"width": 640, "height": 480}
        assert result["input_roi"]["pixel_xyxy"] == [850, 490, 1277, 719]


def test_cpu_preprocess_crops_directly_from_full_frame_without_intermediate_image(
    shared_preprocess_fixture_binary: Path,
) -> None:
    completed = subprocess.run(
        [str(shared_preprocess_fixture_binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert result["image"] == {"width": 8, "height": 6}
    assert result["input_roi"]["pixel_xyxy"] == [2, 1, 6, 5]
    assert result["model_input"] == {"width": 4, "height": 4}
    assert result["first_pixel"] == [2, 1, 3]
    assert result["last_pixel"] == [5, 4, 9]


def test_obb_input_roi_rejects_non_uniform_resize(
    shared_runtime_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "model"
    _write_model_package(package)
    yaml_path = package / "model.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("task_type: detection", "task_type: obb")
    yaml_text = yaml_text.replace("resize_mode: letterbox", "resize_mode: resize")
    yaml_path.write_text(yaml_text, encoding="utf-8")

    with _running_runtime(shared_runtime_binary, ["--model-dir", str(package)]) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        assert status["health"] == "degraded"
        assert "OBB 模型启用 input_roi 时必须使用 letterbox" in status["loaded_model"]["model_load_error"]

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["status"] == "error"
        assert result["error"]["code"] == "MODEL_PACKAGE_INVALID"
