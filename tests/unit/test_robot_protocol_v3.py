"""Unit tests for the production robot protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from production.carton_line.gateway.config import PROJECT_ROOT, load_config
from production.carton_line.gateway.coordinate_mapper import CoordinateMapper
from production.carton_line.gateway.inference_normalizer import normalize_inference_result
from production.carton_line.gateway.register_bank import ProtocolRegisterBank, REG_COORD_BASE


def test_inference_normalizer_keeps_obb_and_v2_fields() -> None:
    result = {
        "image": {"width": 640, "height": 480},
        "detections": [{
            "id": "det-1", "class_id": 0, "class_name": "stand", "score": 0.93,
            "bbox_xyxy": [10, 20, 30, 60], "center_xy": [20, 40],
            "obb": {"w": 20, "h": 40, "angle_deg": -5, "points": [[10, 20], [30, 20], [30, 60], [10, 60]]},
        }],
    }
    payload = normalize_inference_result(result)
    pred = payload["predictions"][0]
    assert payload["image_width"] == 640
    assert pred["confidence"] == 0.93
    assert pred["bbox"] == [10.0, 20.0, 30.0, 60.0]
    assert pred["center"] == [20.0, 40.0]
    assert pred["obb"]["angle_deg"] == -5


def test_protocol_bank_defines_all_200_registers() -> None:
    bank = ProtocolRegisterBank(address_base=0, register_count=200)
    assert len(bank.logical_snapshot()) == 200
    bank.set(101, 3)
    assert bank.get(101) == 3
    assert bank.read(0, 200)[101] == 3


def test_coordinate_mapper_uses_column_major_and_preserves_missing_slots(tmp_path: Path) -> None:
    template = {
        "expected_rows": 2,
        "expected_cols": 2,
        "cells": [
            {"slot_id": 0, "cx": 10, "cy": 20},
            {"slot_id": 1, "cx": 30, "cy": 20},
            {"slot_id": 2, "cx": 10, "cy": 40},
            {"slot_id": 3, "cx": 30, "cy": 40},
        ],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    mapper = CoordinateMapper({
        "template_path": str(template_path),
        "output_frame": "image",
        "register_order": "column",
        "partial_update_enabled": True,
        "partial_match_max_distance_px": 5,
        "partial_min_confidence": 0.1,
        "dual_arm_enabled": False,
    })
    bank = ProtocolRegisterBank()
    bank.set_many(REG_COORD_BASE, [999] * 80)
    result = {"cells": [{"slot_id": 1, "cx": 31, "cy": 21}]}
    updated = mapper.write(bank, result, {"predictions": []})
    values = bank.read(REG_COORD_BASE, 8)
    # slot_id=1 is row=0,col=1 -> column-major register slot index=2.
    assert updated == 1
    assert values[:4] == [999, 999, 999, 999]
    assert values[4:6] == [31, 21]
    assert values[6:8] == [999, 999]


def test_unified_line_config_resolves_all_task_paths() -> None:
    config = load_config(str(PROJECT_ROOT / "production/carton_line/config/line.yaml"))
    assert config["kind"] == "production_line"
    assert Path(config["partition"]["template_path"]).is_file()
    assert Path(config["runtimes"]["partition"]["model_dir"]).is_absolute()
    assert config["collectors"]["partition"]["listen_port"] != config["collectors"]["tube"]["listen_port"]
    assert Path(config["runtimes"]["pick"]["model_dir"]).is_absolute()
    assert config["runtimes"]["pick"]["url"].endswith(":28083")
    assert config["collectors"]["pick"]["listen_port"] == 18093
    assert config["pick"]["tcp"]["server_port"] == 10000


def test_unified_line_config_rejects_duplicate_ports(tmp_path: Path) -> None:
    document = yaml.safe_load((PROJECT_ROOT / "production/carton_line/config/line.yaml").read_text(encoding="utf-8"))
    document["collectors"]["tube"]["listen_port"] = document["collectors"]["partition"]["listen_port"]
    path = tmp_path / "duplicate-port.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="端口必须互不相同"):
        load_config(str(path))
