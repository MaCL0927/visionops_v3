"""carton_tube_check 业务决策测试。"""

from copy import deepcopy

from edge.gateway_adapter.apps.carton_tube_check.decision import evaluate
from edge.gateway_adapter.apps.carton_tube_check.mock_inputs import (
    make_low_confidence_result, make_multi_target_result, make_no_target_result,
    make_ok_result, make_out_of_roi_result, make_size_out_of_range_result,
)
from edge.gateway_adapter.apps.carton_tube_check.register_map import decision_register_values
from edge.gateway_adapter.apps.carton_tube_check.service import DEFAULT_CONFIG


def decide(result):
    return evaluate(result, deepcopy(DEFAULT_CONFIG["rules"]), 7, 1, "example-edge-test")


def test_ok_case_and_details() -> None:
    decision = decide(make_ok_result())
    assert decision.final_label == "OK" and decision.ok
    assert decision.details["target_count"] == 1
    assert decision.details["center_x"] == 640
    assert decision.details["bbox_w"] == 200


def test_no_target_case() -> None:
    assert decide(make_no_target_result()).final_label == "NO_TARGET"


def test_low_confidence_case() -> None:
    assert decide(make_low_confidence_result()).final_label == "LOW_CONFIDENCE"


def test_multi_target_case() -> None:
    assert decide(make_multi_target_result()).final_label == "MULTI_TARGET"


def test_out_of_roi_case() -> None:
    assert decide(make_out_of_roi_result()).final_label == "OUT_OF_ROI"


def test_size_out_of_range_case() -> None:
    assert decide(make_size_out_of_range_result()).final_label == "SIZE_OUT_OF_RANGE"


def test_register_values_follow_decision() -> None:
    values = decision_register_values(decide(make_out_of_roi_result()))
    assert values["final_code"] == 5
    assert values["confidence_x1000"] == 920
    assert values["center_x"] == 1180
    assert values["offset_x_signed"] == 540
