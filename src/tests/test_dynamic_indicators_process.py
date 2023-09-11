import pytest

from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_process import (
    DynamicIndicatorNotExist,
    FtleElementWise,
    FtleGrid,
    LagrangianDescriptor,
    get_dynamic_indicator,
)
from dynamic_indicators_tools.main_dynamic_indicators_process import (
    multi_process_dynamic_indicators,
)


@pytest.fixture
def config_main_test():
    return "tests/config_files/config_main_test_system.json"


@pytest.mark.parametrize(
    "di_method, expected",
    [
        ("ftle_element_wise", FtleElementWise),
        ("ftle_grid", FtleGrid),
        ("lagrangian_descriptors", LagrangianDescriptor),
    ],
)
def test_get_dynamic_indicator(di_method, expected):
    method_object = get_dynamic_indicator(di_method)
    assert isinstance(method_object, expected)


def test_get_dynamic_indicator_error():
    with pytest.raises(DynamicIndicatorNotExist):
        _ = get_dynamic_indicator("fail_method")


def test_main_system_process(config_main_test):
    multi_process_dynamic_indicators(config_main_test)
    assert True
