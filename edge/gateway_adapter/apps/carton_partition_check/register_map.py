"""carton_partition_check 业务寄存器映射。"""

from __future__ import annotations

from edge.modbus_adapter.modbus_registers import RegisterDefinition
from edge.gateway_adapter.apps.common.app_decision import AppDecision


NAMES = (
    "heartbeat", "sequence", "final_code", "ok", "reason_code", "expected_cell_count",
    "cell_count", "missing_count", "defect_count", "max_defect_score_x1000",
    "first_defect_center_x", "first_defect_center_y", "first_defect_bbox_x1",
    "first_defect_bbox_y1", "first_defect_bbox_x2", "first_defect_bbox_y2",
    "frame_id_low", "result_id_low", "error_code", "reserved",
)


def make_register_map(base: int = 200) -> tuple[RegisterDefinition, ...]:
    return tuple(
        RegisterDefinition(base + index, name, "uint16", 1.0, f"隔板业务 {name}")
        for index, name in enumerate(NAMES)
    )


def decision_register_values(decision: AppDecision) -> dict[str, int]:
    details = decision.details
    return {
        "heartbeat": decision.heartbeat,
        "sequence": decision.sequence & 0xFFFF,
        "final_code": decision.final_code,
        "ok": int(decision.ok),
        "reason_code": decision.reason_code,
        "expected_cell_count": int(details.get("expected_cell_count", 0)),
        "cell_count": int(details.get("cell_count", 0)),
        "missing_count": int(details.get("missing_count", 0)),
        "defect_count": int(details.get("defect_count", 0)),
        "max_defect_score_x1000": int(details.get("max_defect_score_x1000", 0)),
        "first_defect_center_x": int(details.get("first_defect_center_x", 0)),
        "first_defect_center_y": int(details.get("first_defect_center_y", 0)),
        "first_defect_bbox_x1": int(details.get("first_defect_bbox_x1", 0)),
        "first_defect_bbox_y1": int(details.get("first_defect_bbox_y1", 0)),
        "first_defect_bbox_x2": int(details.get("first_defect_bbox_x2", 0)),
        "first_defect_bbox_y2": int(details.get("first_defect_bbox_y2", 0)),
        "frame_id_low": int(details.get("frame_id_low", 0)),
        "result_id_low": int(details.get("result_id_low", 0)),
        "error_code": int(details.get("error_code", 0)),
        "reserved": 0,
    }
