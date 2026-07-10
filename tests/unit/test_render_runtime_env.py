"""runtime env 渲染器的单元测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tools.config.render_runtime_env import render_runtime_env, write_atomic
from tools.config.validate_config import load_configuration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGE_FILES = [
    PROJECT_ROOT / "configs/edge/base.example.yaml",
    PROJECT_ROOT / "configs/edge/rk3576.example.yaml",
]
TASK_FILE = PROJECT_ROOT / "configs/task/roi_classification.example.yaml"
APP_FILES = [PROJECT_ROOT / "configs/app/collector.example.yaml"]


def test_rendered_env_contains_metadata_and_key_values(tmp_path: Path) -> None:
    edge, task, apps = load_configuration(EDGE_FILES, TASK_FILE, APP_FILES)
    source_paths = [*EDGE_FILES, TASK_FILE, *APP_FILES]

    content = render_runtime_env(
        edge,
        task,
        apps,
        source_paths,
        generated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )
    output = tmp_path / "visionops-runtime.env"
    write_atomic(output, content)

    rendered = output.read_text(encoding="utf-8")
    assert 'VISIONOPS_CONFIG_SCHEMA_VERSION="1.0"' in rendered
    assert 'VISIONOPS_CONFIG_SOURCE_SHA256="' in rendered
    assert 'VISIONOPS_CONFIG_GENERATED_AT="2026-01-02T03:04:05Z"' in rendered
    assert 'VISIONOPS_EDGE_DEVICE_PLATFORM="rk3576"' in rendered
    assert 'VISIONOPS_TASK_TASK_TYPE="roi_classification"' in rendered
    assert 'VISIONOPS_APP_COLLECTOR_APP_NAME="collector"' in rendered
    assert str(EDGE_FILES[0].resolve()) in rendered
