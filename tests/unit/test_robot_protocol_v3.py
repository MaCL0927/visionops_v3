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
    assert config["pick"]["websocket"]["listen_port"] == 9001
    assert config["pick"]["websocket"]["path"] == "/vision"
    assert config["pick"]["video"]["type"] == "mjpeg"


def test_unified_line_config_rejects_duplicate_ports(tmp_path: Path) -> None:
    document = yaml.safe_load((PROJECT_ROOT / "production/carton_line/config/line.yaml").read_text(encoding="utf-8"))
    document["collectors"]["tube"]["listen_port"] = document["collectors"]["partition"]["listen_port"]
    path = tmp_path / "duplicate-port.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="端口必须互不相同"):
        load_config(str(path))


def test_coordinate_mapper_selects_four_zone_affines(tmp_path: Path) -> None:
    template = {
        "expected_rows": 5,
        "expected_cols": 8,
        "cells": [
            {"slot_id": sid, "cx": sid % 8, "cy": sid // 8}
            for sid in range(40)
        ],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    def affine(b0: int, b1: int) -> dict[str, float]:
        return {"a00": 1.0, "a01": 0.0, "a10": 0.0, "a11": 1.0, "b0": b0, "b1": b1}

    mapper = CoordinateMapper({
        "template_path": str(template_path),
        "output_frame": "robot",
        "register_order": "row",
        "partial_update_enabled": False,
        "dual_arm_enabled": True,
        "four_zone_enabled": True,
        "left_columns": [0, 3],
        "right_columns": [4, 7],
        "top_rows": [0, 2],
        "bottom_rows": [3, 4],
        "left_affine": affine(1000, 1000),
        "right_affine": affine(2000, 2000),
        "left_top_affine": affine(10, 20),
        "left_bottom_affine": affine(30, 40),
        "right_top_affine": affine(50, 60),
        "right_bottom_affine": affine(70, 80),
    })
    bank = ProtocolRegisterBank()
    result = {"cells": [
        {"slot_id": 0, "cx": 1, "cy": 2},    # left/top
        {"slot_id": 24, "cx": 3, "cy": 4},  # left/bottom
        {"slot_id": 4, "cx": 5, "cy": 6},   # right/top
        {"slot_id": 28, "cx": 7, "cy": 8},  # right/bottom
    ]}
    assert mapper.write(bank, result, {"predictions": []}) == 4

    by_slot = {int(cell["slot_id"]): cell for cell in result["cells"]}
    assert (by_slot[0]["robot_cx"], by_slot[0]["robot_cy"]) == (11, 22)
    assert by_slot[0]["coord_transform_key"] == "left_top_affine"
    assert (by_slot[24]["robot_cx"], by_slot[24]["robot_cy"]) == (33, 44)
    assert by_slot[24]["coord_transform_key"] == "left_bottom_affine"
    assert (by_slot[4]["robot_cx"], by_slot[4]["robot_cy"]) == (55, 66)
    assert by_slot[4]["coord_transform_key"] == "right_top_affine"
    assert (by_slot[28]["robot_cx"], by_slot[28]["robot_cy"]) == (77, 88)
    assert by_slot[28]["coord_transform_key"] == "right_bottom_affine"
    assert result["coordinate_update"]["four_zone_enabled"] is True


def test_coordinate_mapper_four_zone_missing_matrix_falls_back_to_arm_affine(tmp_path: Path) -> None:
    template = {
        "expected_rows": 5,
        "expected_cols": 8,
        "cells": [{"slot_id": 24, "cx": 3, "cy": 4}],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    mapper = CoordinateMapper({
        "template_path": str(template_path),
        "output_frame": "robot",
        "register_order": "row",
        "partial_update_enabled": False,
        "dual_arm_enabled": True,
        "four_zone_enabled": True,
        "left_columns": [0, 3],
        "right_columns": [4, 7],
        "top_rows": [0, 2],
        "bottom_rows": [3, 4],
        "left_affine": {"a00": 1, "a01": 0, "a10": 0, "a11": 1, "b0": 100, "b1": 200},
    })
    bank = ProtocolRegisterBank()
    result = {"cells": [{"slot_id": 24, "cx": 3, "cy": 4}]}
    mapper.write(bank, result, {"predictions": []})
    cell = result["cells"][0]
    assert (cell["robot_cx"], cell["robot_cy"]) == (103, 204)
    assert cell["coord_transform_key"] == "left_affine"
