from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = PROJECT_ROOT / "scripts" / "start_runtime.sh"
VERIFY_SCRIPT = PROJECT_ROOT / "production" / "foam_ring_grasp" / "scripts" / "verify_rgb_runtime.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("m36_2_verify", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(path: Path, model: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "active_camera": model,
                "cameras": {
                    "orbbec336l": {
                        "base_url": "http://127.0.0.1:18182",
                        "snapshot_path": "/stream/snapshot.jpg",
                        "health_path": "/health",
                    },
                    "hp60c": {
                        "base_url": "http://127.0.0.1:18181",
                        "snapshot_path": "/stream/snapshot.jpg",
                        "health_path": "/health",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _run_start_script(tmp_path: Path, camera_model: str) -> list[str]:
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python3").symlink_to(Path(sys.executable))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.rknn").write_bytes(b"fixture")
    (model / "model.yaml").write_text("task: segmentation\n", encoding="utf-8")
    selection = tmp_path / "active_camera.json"
    _selection(selection, camera_model)
    args_file = tmp_path / "args.json"
    runtime = tmp_path / "runtime"
    runtime.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['ARGS_FILE'], 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    runtime.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "VISIONOPS_EDGE_ROOT": str(PROJECT_ROOT),
            "VISIONOPS_VENV": str(tmp_path / "venv"),
            "VISIONOPS_RUNTIME_BIN": str(runtime),
            "VISIONOPS_CAMERA_SELECTION_FILE": str(selection),
            "VISIONOPS_FRAME_SOURCE": "auto",
            "ARGS_FILE": str(args_file),
        }
    )
    subprocess.run(
        ["bash", str(START_SCRIPT), str(model)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(args_file.read_text(encoding="utf-8"))


def _value_after(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_start_runtime_uses_shared_memory_for_orbbec(tmp_path: Path) -> None:
    args = _run_start_script(tmp_path, "orbbec336l")
    assert _value_after(args, "--frame-source") == "shared_memory"
    assert _value_after(args, "--shared-memory-name") == "/visionops_orbbec336l_rgb"
    assert _value_after(args, "--hp60c-url") == "http://127.0.0.1:18182"


def test_start_runtime_keeps_http_bridge_for_hp60c(tmp_path: Path) -> None:
    args = _run_start_script(tmp_path, "hp60c")
    assert _value_after(args, "--frame-source") == "hp60c_bridge"
    assert _value_after(args, "--hp60c-url") == "http://127.0.0.1:18181"


def test_m36_2_timing_summary() -> None:
    module = _load_verify_module()
    summary = module._timing_summary([10.0, 20.0, 30.0, 40.0])
    assert summary == {"mean": 25.0, "p50": 20.0, "p95": 40.0, "max": 40.0}
