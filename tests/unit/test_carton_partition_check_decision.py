"""carton_partition_check 业务决策测试。"""

from copy import deepcopy

from edge.gateway_adapter.apps.carton_partition_check.decision import evaluate
from edge.gateway_adapter.apps.carton_partition_check.mock_inputs import (
    make_defect_result, make_low_confidence_result, make_missing_cell_result,
    make_no_target_result, make_ok_result,
)
from edge.gateway_adapter.apps.carton_partition_check.register_map import decision_register_values
from edge.gateway_adapter.apps.carton_partition_check.service import DEFAULT_CONFIG


def decide(result, rules=None):
    return evaluate(result, rules or deepcopy(DEFAULT_CONFIG["rules"]), 9, 1, "example-edge-test")


def test_ok_case() -> None:
    decision = decide(make_ok_result())
    assert decision.final_label == "OK" and decision.ok
    assert decision.details["cell_count"] == 12


def test_no_target_case() -> None:
    assert decide(make_no_target_result()).final_label == "NO_TARGET"


def test_missing_cell_case() -> None:
    decision = decide(make_missing_cell_result())
    assert decision.final_label == "STRUCTURE_ABNORMAL"
    assert decision.details["missing_count"] == 2


def test_defect_case_and_priority() -> None:
    result = make_missing_cell_result()
    result["detections"].append(make_defect_result()["detections"][-1])
    decision = decide(result)
    assert decision.final_label == "STRUCTURE_ABNORMAL"
    assert decision.reason == "检测到隔板缺陷目标"
    assert decision.details["defect_count"] == 1


def test_low_confidence_case() -> None:
    assert decide(make_low_confidence_result()).final_label == "LOW_CONFIDENCE"


def test_expected_cell_count_has_priority_over_range() -> None:
    rules = deepcopy(DEFAULT_CONFIG["rules"])
    rules["expected_cell_count"] = 11
    rules["min_cell_count"] = 1
    rules["max_cell_count"] = 20
    assert decide(make_ok_result(), rules).final_label == "STRUCTURE_ABNORMAL"


def test_register_values_follow_defect() -> None:
    values = decision_register_values(decide(make_defect_result()))
    assert values["final_code"] == 7
    assert values["defect_count"] == 1
    assert values["max_defect_score_x1000"] == 880
    assert values["first_defect_center_x"] == 590
