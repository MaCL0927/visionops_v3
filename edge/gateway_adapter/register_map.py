"""Gateway Mock 默认 Holding Register 业务映射。"""

from __future__ import annotations

from edge.modbus_adapter.modbus_registers import RegisterDefinition


DEFAULT_REGISTER_MAP: tuple[RegisterDefinition, ...] = (
    RegisterDefinition(0, "heartbeat", "uint16", 1.0, "Gateway 心跳翻转值"),
    RegisterDefinition(1, "sequence", "uint16", 1.0, "Gateway 消息序号低 16 位"),
    RegisterDefinition(2, "status_code", "uint16", 1.0, "0 表示成功，1 表示错误"),
    RegisterDefinition(3, "final_code", "uint16", 1.0, "最终业务代码的 16 位表示"),
    RegisterDefinition(4, "ok", "uint16", 1.0, "业务结果是否通过，1 为 true"),
    RegisterDefinition(5, "reason_code", "uint16", 1.0, "原因文本的稳定 16 位编码"),
    RegisterDefinition(6, "error_code", "uint16", 1.0, "错误代码的稳定 16 位编码"),
    RegisterDefinition(7, "object_count", "uint16", 1.0, "检测对象数量"),
    RegisterDefinition(8, "score_x1000", "uint16", 1.0, "最高置信度乘以 1000"),
    RegisterDefinition(9, "center_x", "uint16", 1.0, "最高分目标中心 X 像素"),
    RegisterDefinition(10, "center_y", "uint16", 1.0, "最高分目标中心 Y 像素"),
    RegisterDefinition(11, "bbox_x1", "uint16", 1.0, "最高分目标框左上 X"),
    RegisterDefinition(12, "bbox_y1", "uint16", 1.0, "最高分目标框左上 Y"),
    RegisterDefinition(13, "bbox_x2", "uint16", 1.0, "最高分目标框右下 X"),
    RegisterDefinition(14, "bbox_y2", "uint16", 1.0, "最高分目标框右下 Y"),
    RegisterDefinition(15, "inference_ms", "uint16", 1.0, "推理耗时毫秒"),
    RegisterDefinition(16, "total_ms", "uint16", 1.0, "总耗时毫秒"),
    RegisterDefinition(17, "frame_id_low", "uint16", 1.0, "Frame ID 的稳定低 16 位"),
    RegisterDefinition(18, "result_id_low", "uint16", 1.0, "Result ID 的稳定低 16 位"),
    RegisterDefinition(19, "reserved", "uint16", 1.0, "预留寄存器"),
)


def definitions_by_name() -> dict[str, RegisterDefinition]:
    return {definition.name: definition for definition in DEFAULT_REGISTER_MAP}
