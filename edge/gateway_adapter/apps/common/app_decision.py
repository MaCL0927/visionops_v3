"""业务 App 统一决策结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

from edge.gateway_adapter.gateway_message import timestamp_ms


class FinalCode(IntEnum):
    OK = 0
    NG = 1
    NO_TARGET = 2
    MULTI_TARGET = 3
    LOW_CONFIDENCE = 4
    OUT_OF_ROI = 5
    SIZE_OUT_OF_RANGE = 6
    STRUCTURE_ABNORMAL = 7
    UPSTREAM_NO_RESULT = 8
    INTERNAL_ERROR = 9


@dataclass(frozen=True)
class AppDecision:
    app_id: str
    device_id: str
    frame_id: str
    result_id: str
    sequence: int
    heartbeat: int
    final_code: int
    final_label: str
    ok: bool
    reason_code: int
    reason: str
    object_count: int
    confidence_x1000: int
    primary_target: dict[str, Any] | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    timestamp_ms: int = field(default_factory=timestamp_ms)
    schema_version: str = "1.0"
    message_type: str = "app_decision"

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        if self.primary_target is None:
            document.pop("primary_target")
        if not self.measurements:
            document.pop("measurements")
        if not self.details:
            document.pop("details")
        if self.error is None:
            document.pop("error")
        return document
