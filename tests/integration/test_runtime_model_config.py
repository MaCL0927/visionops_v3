"""Runtime 模型包与轻量配置读取的无设备集成测试。"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PACKAGE = PROJECT_ROOT / "edge/runtime_cpp/examples/mock_model_package"


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
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture(scope="session")
def model_config_runtime_binary(shared_runtime_binary: Path) -> Path:
    return shared_runtime_binary

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
            "example-model-config-test",
            *extra_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
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


def test_runtime_uses_default_mock_model_without_config(
    model_config_runtime_binary: Path,
) -> None:
    with _running_runtime(model_config_runtime_binary, []) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert status["health"] == "ok"
        assert model["model_id"] == "model-mock-001"
        assert model["model_name"] == "visionops-runtime-mock"
        assert model["task_type"] == "detection"
        assert model["backend"] == "mock"
        assert model["model_load_error"] is None
        # Global compatibility default: non-segmentation and legacy packages do
        # not opt into the high-resolution segmentation decoder implicitly.
        assert model["mask_decode_mode"] == "legacy_proto"


def test_runtime_reads_m15_model_dir(
    model_config_runtime_binary: Path,
) -> None:
    with _running_runtime(
        model_config_runtime_binary,
        [
            "--model-dir",
            str(EXAMPLE_PACKAGE),
        ],
    ) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert status["health"] == "ok"
        assert model["model_id"] == "package-mock-tube-detector-2.1.0-rk3588"
        assert model["model_name"] == "mock-tube-detector-yaml"
        assert model["model_version"] == "2.1.1"
        assert model["task_type"] == "detection"
        assert model["target_platform"] == "rk3588"
        assert model["rknn_path"].endswith("mock_model_package/model.rknn")
        assert model["config_path"].endswith("mock_model_package/model.yaml")
        assert model["labels_count"] == 2
        assert model["input_size"] == {"width": 768, "height": 768}
        assert model["score_threshold"] == pytest.approx(0.55)
        assert model["nms_threshold"] == pytest.approx(0.42)

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["task_type"] == "detection"
        assert result["model"]["model_id"] == model["model_id"]
        assert result["model"]["model_name"] == model["model_name"]
        assert result["model"]["input_size"] == model["input_size"]
        assert result["model"]["labels_count"] == 2


def test_missing_model_files_degrade_without_stopping_runtime(
    model_config_runtime_binary: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken_model"
    broken.mkdir()
    with _running_runtime(
        model_config_runtime_binary,
        [
            "--model-dir",
            str(broken),
        ],
    ) as base_url:
        health = _request_json(f"{base_url}/health")
        status = _request_json(f"{base_url}/api/runtime/status")
        assert health["health"] == "degraded"
        assert status["health"] == "degraded"
        assert "缺少 model.rknn" in status["loaded_model"]["model_load_error"]
        assert "缺少 model.yaml" in status["loaded_model"]["model_load_error"]

        result = _request_json(f"{base_url}/api/runtime/infer_once", method="POST")
        assert result["message_type"] == "inference_result"
        assert result["model"]["model_name"] == "broken_model"


def test_runtime_reads_postprocess_limits_and_cli_overrides(
    model_config_runtime_binary: Path, tmp_path: Path
) -> None:
    package = tmp_path / "limited_segmentation"
    package.mkdir()
    (package / "model.rknn").write_bytes(b"mock")
    (package / "model.yaml").write_text(
        "\n".join(
            [
                'model_id: limited-segmentation',
                'model_name: limited-segmentation',
                'version: 1.0.0',
                'task: segmentation',
                'target_platform: rk3576',
                'input_size: [640, 640]',
                'class_names: [box]',
                'max_detections: 3',
                'mask_max_points: 48',
                'mask_decode_mode: legacy_proto',
                'mask_threshold: 0.55',
                'mask_polygon_epsilon_px: 1.25',
            ]
        ),
        encoding="utf-8",
    )

    with _running_runtime(
        model_config_runtime_binary,
        [
            "--model-dir",
            str(package),
            "--max-detections",
            "1",
            "--mask-max-points",
            "32",
        ],
    ) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert model["max_detections"] == 1
        assert model["mask_max_points"] == 32
        assert model["mask_decode_mode"] == "legacy_proto"
        assert model["mask_threshold"] == pytest.approx(0.55)
        assert model["mask_polygon_epsilon_px"] == pytest.approx(1.25)


def test_runtime_accepts_pyyaml_block_lists_for_input_roi(
    model_config_runtime_binary: Path, tmp_path: Path
) -> None:
    """兼容历史设置页 safe_dump 产生的 input ROI 块列表。"""

    package = tmp_path / "block_list_roi_model"
    package.mkdir()
    (package / "model.rknn").write_bytes(b"mock")
    (package / "model.yaml").write_text(
        """schema_version: '1.0'
model_id: block-list-roi
model_name: block-list-roi
model_version: 20260725165208
task_type: detection
target_platform: rk3576
input_size:
- 640
- 640
class_names:
- blue
postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
preprocess:
  input_roi:
    enabled: true
    coordinate_space: runtime_snapshot
    source_resolution:
      width: 1280
      height: 720
    pixel_xyxy:
    - 850
    - 490
    - 1277
    - 719
    normalized_xyxy:
    - 0.6640625
    - 0.680555555556
    - 0.99765625
    - 0.998611111111
    crop_resolution:
      width: 427
      height: 229
    resize_mode: letterbox
    pad_value: 114
""",
        encoding="utf-8",
    )

    with _running_runtime(
        model_config_runtime_binary,
        ["--model-dir", str(package)],
    ) as base_url:
        status = _request_json(f"{base_url}/api/runtime/status")
        model = status["loaded_model"]
        assert status["health"] == "ok"
        assert model["model_load_error"] is None
        assert model["input_size"] == {"width": 640, "height": 640}
        assert model["input_roi"]["enabled"] is True
        assert model["input_roi"]["pixel_xyxy"] == [850, 490, 1277, 719]
        assert model["input_roi"]["normalized_xyxy"] == pytest.approx(
            [0.6640625, 0.680555555556, 0.99765625, 0.998611111111]
        )
