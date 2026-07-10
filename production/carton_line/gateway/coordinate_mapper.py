"""Partition-cell coordinate mapping for trigger register 103."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .register_bank import ProtocolRegisterBank, REG_COORD_BASE


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slot_id(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def _encode_int16(value: int) -> int:
    return int(value) & 0xFFFF


class CoordinateMapper:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.template_path = Path(str(self.config.get("template_path", "")))
        self.output_frame = str(self.config.get("output_frame", "image")).lower()
        self.register_order = str(self.config.get("register_order", "column")).lower()
        self.partial_update_enabled = bool(self.config.get("partial_update_enabled", True))
        self.partial_match_max_distance_px = float(self.config.get("partial_match_max_distance_px", 22.0))
        self.partial_min_confidence = float(self.config.get("partial_min_confidence", 0.1))
        self.dual_arm_enabled = bool(self.config.get("dual_arm_enabled", False))
        self.left_columns = tuple(int(x) for x in self.config.get("left_columns", [0, 3]))
        self.right_columns = tuple(int(x) for x in self.config.get("right_columns", [4, 7]))

    def always_ok(self) -> bool:
        return bool(self.config.get("always_ok", True))

    def _load_template(self) -> tuple[int, int, list[dict[str, Any]]]:
        document = json.loads(self.template_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("坐标模板顶层必须是对象")
        rows = int(document.get("expected_rows") or 5)
        cols = int(document.get("expected_cols") or 8)
        cells = document.get("cells") if isinstance(document.get("cells"), list) else []
        parsed = []
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            sid = _slot_id(cell.get("slot_id"))
            cx, cy = _float(cell.get("cx")), _float(cell.get("cy"))
            if sid is None:
                sid = index
            if cx is None or cy is None or not 0 <= sid < rows * cols:
                continue
            parsed.append({"slot_id": sid, "cx": cx, "cy": cy})
        if not parsed:
            raise ValueError(f"坐标模板没有有效 cells: {self.template_path}")
        return rows, cols, parsed

    @staticmethod
    def _valid_cells(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        cells = result.get("cells") if isinstance(result.get("cells"), list) else []
        valid = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            sid = _slot_id(cell.get("slot_id"))
            cx, cy = _float(cell.get("cx")), _float(cell.get("cy"))
            if sid is not None and cx is not None and cy is not None:
                valid.append(cell)
        return valid

    def _partial_cells(self, payload: Mapping[str, Any], template: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        predictions = payload.get("predictions") if isinstance(payload.get("predictions"), list) else []
        best_by_slot: dict[int, dict[str, Any]] = {}
        filtered = 0
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            confidence = _float(pred.get("confidence")) or 0.0
            center = pred.get("center")
            bbox = pred.get("bbox")
            if confidence < self.partial_min_confidence:
                filtered += 1
                continue
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = _float(center[0]), _float(center[1])
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = [_float(x) for x in bbox[:4]]
                if None in {x1, y1, x2, y2}:
                    cx = cy = None
                else:
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0  # type: ignore[operator]
            else:
                cx = cy = None
            if cx is None or cy is None:
                filtered += 1
                continue
            nearest = min(template, key=lambda item: (cx - float(item["cx"])) ** 2 + (cy - float(item["cy"])) ** 2)
            distance = math.hypot(cx - float(nearest["cx"]), cy - float(nearest["cy"]))
            if distance > self.partial_match_max_distance_px:
                filtered += 1
                continue
            sid = int(nearest["slot_id"])
            cell = {
                "slot_id": sid,
                "cx": cx,
                "cy": cy,
                "bbox": list(bbox[:4]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else None,
                "confidence": confidence,
                "partial_slot_match_dist_px": round(distance, 3),
                "partial_update_source": "runtime_prediction_nearest_template",
            }
            previous = best_by_slot.get(sid)
            if previous is None or distance < float(previous["partial_slot_match_dist_px"]):
                best_by_slot[sid] = cell
        cells = [best_by_slot[key] for key in sorted(best_by_slot)]
        return cells, {
            "enabled": True,
            "source": "runtime_predictions_nearest_template",
            "template_path": str(self.template_path),
            "raw_prediction_count": len(predictions),
            "matched_cell_count": len(cells),
            "unmatched_or_filtered_count": filtered,
            "min_conf": self.partial_min_confidence,
            "max_match_dist_px": self.partial_match_max_distance_px,
        }

    def ensure_cells(self, result: dict[str, Any], payload: Mapping[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
        rows, cols, template = self._load_template()
        existing = self._valid_cells(result)
        if existing:
            result["coord_partial_update_debug"] = {
                "enabled": self.partial_update_enabled,
                "source": "existing_analyze_cells",
                "matched_cell_count": len(existing),
            }
            return rows, cols, existing
        if not self.partial_update_enabled:
            return rows, cols, []
        cells, debug = self._partial_cells(payload, template)
        result["coord_partial_update_debug"] = debug
        if cells:
            result["cells"] = cells
            result["valid_cell_count"] = len(cells)
            result["coord_cells_filled_from_runtime_predictions"] = True
        return rows, cols, cells

    def _register_index(self, sid: int, rows: int, cols: int) -> int | None:
        if not 0 <= sid < rows * cols:
            return None
        if self.register_order.replace("-", "_") in {
            "column", "col", "column_major", "col_major", "down_then_right", "top_down_left_right"
        }:
            row, col = divmod(sid, cols)
            return col * rows + row
        return sid

    def _arm(self, sid: int, cols: int) -> str:
        if not self.dual_arm_enabled:
            return "single"
        col = sid % cols
        if self.left_columns[0] <= col <= self.left_columns[1]:
            return "left"
        if self.right_columns[0] <= col <= self.right_columns[1]:
            return "right"
        return "left" if col < cols / 2.0 else "right"

    def _transform(self, x: float, y: float, arm: str) -> tuple[int, int]:
        if self.output_frame not in {"robot", "robot_mm", "base", "robot_base"}:
            return _int16(x), _int16(y)
        key = f"{arm}_affine" if arm in {"left", "right"} else "single_affine"
        affine = self.config.get(key) if isinstance(self.config.get(key), Mapping) else {}
        a00 = float(affine.get("a00", 1.0)); a01 = float(affine.get("a01", 0.0))
        a10 = float(affine.get("a10", 0.0)); a11 = float(affine.get("a11", 1.0))
        b0 = float(affine.get("b0", 0.0)); b1 = float(affine.get("b1", 0.0))
        return _int16(a00 * x + a01 * y + b0), _int16(a10 * x + a11 * y + b1)

    def write(self, bank: ProtocolRegisterBank, result: dict[str, Any], payload: Mapping[str, Any]) -> int:
        rows, cols, cells = self.ensure_cells(result, payload)
        coordinates = bank.read(bank.address_base + REG_COORD_BASE, 80)
        updated = 0
        for cell in cells:
            sid = _slot_id(cell.get("slot_id"))
            cx, cy = _float(cell.get("cx")), _float(cell.get("cy"))
            if sid is None or cx is None or cy is None:
                continue
            index = self._register_index(sid, rows, cols)
            if index is None or not 0 <= index < 40:
                continue
            arm = self._arm(sid, cols)
            out_x, out_y = self._transform(cx, cy, arm)
            coordinates[index * 2] = _encode_int16(out_x)
            coordinates[index * 2 + 1] = _encode_int16(out_y)
            cell.update({
                "output_frame": self.output_frame,
                "coord_arm": arm,
                "register_order": self.register_order,
                "vision_slot_id": sid,
                "register_slot_id": index,
                "register_x": REG_COORD_BASE + index * 2,
                "register_y": REG_COORD_BASE + index * 2 + 1,
                "image_cx": cx,
                "image_cy": cy,
                "robot_cx": out_x,
                "robot_cy": out_y,
            })
            updated += 1
        bank.set_many(REG_COORD_BASE, coordinates)
        result["coordinate_update"] = {
            "updated_slots": updated,
            "preserved_slots": 40 - updated,
            "output_frame": self.output_frame,
            "register_order": self.register_order,
        }
        return updated
