"""算法设置保存不得重排标准模型 YAML。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apps.collector_web.backend.algorithm_settings import (
    apply_algorithm_settings,
    get_algorithm_settings_payload,
)


BASE_TEMPLATE = """schema_version: '1.0'
model_id: {model_id}
model_name: test-{task}
model_version: 20260725_165208
task_type: {task}
target_platform: rk3576
input_size: [640, 640]
model:
  name: test-{task}
  version: 20260725_165208
  task: {task}
  format: rknn
  target_platform: rk3576
  input_size: [640, 640]
classes:
- id: 0
  name: blue
class_names:
- blue
postprocess:
  conf_threshold: 0.25  # keep score comment
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
"""


def _make_package(root: Path, task: str) -> Path:
    package = root / f"model-{task}"
    package.mkdir(parents=True)
    (package / "model.rknn").write_bytes(b"rknn")
    text = BASE_TEMPLATE.format(model_id=f"model-{task}", task=task)
    if task == "classification":
        text = text.replace("  iou_threshold: 0.45\n", "")
    (package / "model.yaml").write_text(text, encoding="utf-8")
    return package


@pytest.mark.parametrize("task", ["detection", "segmentation", "obb"])
def test_threshold_update_preserves_yaml_layout_for_nms_tasks(tmp_path: Path, task: str) -> None:
    models_root = tmp_path / "models"
    package = _make_package(models_root, task)
    yaml_path = package / "model.yaml"
    before = yaml_path.read_text(encoding="utf-8")

    result = apply_algorithm_settings(
        models_root,
        {
            "model_id": f"model-{task}",
            "score_threshold": 0.37,
            "nms_threshold": 0.52,
            "reload_runtime": False,
        },
    )

    after = yaml_path.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert "  conf_threshold: 0.37  # keep score comment" in after
    assert "  iou_threshold: 0.52" in after

    # 这些行在旧实现中会被 safe_dump 改写。
    for unchanged in [
        "model_version: 20260725_165208",
        "input_size: [640, 640]",
        "  input_size: [640, 640]",
        "    pixel_xyxy: [850, 490, 1277, 719]",
        "    normalized_xyxy: [0.6640625, 0.680555555556, 0.99765625, 0.998611111111]",
    ]:
        assert unchanged in before
        assert unchanged in after

    assert "\nscore_threshold:" not in after
    assert "\nnms_threshold:" not in after
    parsed = yaml.safe_load(after)
    assert parsed["postprocess"]["conf_threshold"] == pytest.approx(0.37)
    assert parsed["postprocess"]["iou_threshold"] == pytest.approx(0.52)


def test_classification_updates_score_only_and_preserves_yaml(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    package = _make_package(models_root, "classification")
    yaml_path = package / "model.yaml"

    result = apply_algorithm_settings(
        models_root,
        {
            "model_id": "model-classification",
            "score_threshold": 0.61,
            "nms_threshold": 0.11,
        },
    )
    after = yaml_path.read_text(encoding="utf-8")

    assert result["changed"] is True
    assert "  conf_threshold: 0.61  # keep score comment" in after
    assert "iou_threshold" not in after
    assert "nms_threshold" not in after
    assert "    pixel_xyxy: [850, 490, 1277, 719]" in after


def test_settings_read_prefers_postprocess_and_repairs_legacy_duplicates(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    package = _make_package(models_root, "detection")
    yaml_path = package / "model.yaml"
    with yaml_path.open("a", encoding="utf-8") as handle:
        handle.write("score_threshold: 0.8\nnms_threshold: 0.7\n")

    payload = get_algorithm_settings_payload(models_root, model_id="model-detection")
    assert payload["settings"]["score_threshold"] == pytest.approx(0.25)
    assert payload["settings"]["score_threshold_key"] == "postprocess.conf_threshold"
    assert payload["settings"]["nms_threshold"] == pytest.approx(0.45)

    apply_algorithm_settings(
        models_root,
        {"model_id": "model-detection", "score_threshold": 0.33, "nms_threshold": 0.44},
    )
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert parsed["postprocess"]["conf_threshold"] == pytest.approx(0.33)
    assert parsed["score_threshold"] == pytest.approx(0.33)
    assert parsed["postprocess"]["iou_threshold"] == pytest.approx(0.44)
    assert parsed["nms_threshold"] == pytest.approx(0.44)


def test_missing_postprocess_thresholds_are_inserted_without_reformat(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    package = _make_package(models_root, "detection")
    yaml_path = package / "model.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("  conf_threshold: 0.25  # keep score comment\n", "")
    text = text.replace("  iou_threshold: 0.45\n", "")
    yaml_path.write_text(text, encoding="utf-8")

    apply_algorithm_settings(
        models_root,
        {"model_id": "model-detection", "score_threshold": 0.29, "nms_threshold": 0.41},
    )
    after = yaml_path.read_text(encoding="utf-8")
    assert "  conf_threshold: 0.29" in after
    assert "  iou_threshold: 0.41" in after
    assert "    pixel_xyxy: [850, 490, 1277, 719]" in after
