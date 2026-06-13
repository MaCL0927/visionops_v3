"""M6 业务 register map 和配置契约测试。"""

from pathlib import Path

import pytest

from edge.gateway_adapter.apps.carton_partition_check.register_map import make_register_map as partition_map
from edge.gateway_adapter.apps.carton_tube_check.register_map import make_register_map as tube_map
from edge.gateway_adapter.apps.common.app_register_bank import AppRegisterBank, decode_int16, encode_int16
from tools.config.validate_config import load_configuration, validate_configuration


ROOT = Path(__file__).resolve().parents[2]


def test_business_register_maps_do_not_overlap() -> None:
    tube = {item.address for item in tube_map()}
    partition = {item.address for item in partition_map()}
    assert tube == set(range(100, 120))
    assert partition == set(range(200, 220))
    assert tube.isdisjoint(partition)


def test_all_definitions_are_complete() -> None:
    for item in (*tube_map(), *partition_map()):
        assert isinstance(item.address, int)
        assert item.name and item.data_type and item.description
        assert item.scale > 0


@pytest.mark.parametrize("value", [-32768, -6, 0, 7, 32767])
def test_signed_int16_encoding_is_stable(value: int) -> None:
    assert decode_int16(encode_int16(value)) == value


def test_uint16_range_is_enforced() -> None:
    bank = AppRegisterBank(tube_map())
    with pytest.raises(ValueError):
        bank.write(100, [65536])
    with pytest.raises(ValueError):
        encode_int16(32768)


def test_business_example_configs_pass_m1_validation() -> None:
    edge, task, apps = load_configuration(
        [ROOT / "configs/edge/base.example.yaml", ROOT / "configs/edge/rk3588.example.yaml"],
        ROOT / "configs/task/detection.example.yaml",
        [
            ROOT / "configs/app/carton_tube_check.example.yaml",
            ROOT / "configs/app/carton_partition_check.example.yaml",
        ],
    )
    validate_configuration(edge, task, apps)
