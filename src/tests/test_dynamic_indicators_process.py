import json
import os

import pytest

from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_utils import (
    DynamicIndicatorNotExist,
    FtleElementWise,
    FtleGrid,
    LagrangianDescriptor,
    get_dynamic_indicator,
)


@pytest.fixture
def config_main_test():
    return "tests/config_files/config_main_test_system.json"


@pytest.mark.parametrize(
    "di_method, expected",
    [
        (FtleElementWise.name_dynamic_indicator, FtleElementWise),
        (FtleGrid.name_dynamic_indicator, FtleGrid),
        (LagrangianDescriptor.name_dynamic_indicator, LagrangianDescriptor),
        # ("poincare_section", PoincareSections),
    ],
)
def test_get_dynamic_indicator(di_method, expected):
    method_object = get_dynamic_indicator(di_method)
    assert isinstance(method_object, expected)


def test_get_dynamic_indicator_error():
    with pytest.raises(DynamicIndicatorNotExist):
        _ = get_dynamic_indicator("fail_method")


def test_main_process(config_main_test):
    with open(config_main_test, "r") as file_json:
        params = json.load(file_json)
    dynamic_indicators = params.copy()
    system_params = dynamic_indicators.pop("system_params")
    path = system_params.get("path")
    if os.path.exists(path):
        for file_path in os.scandir(path):
            os.remove(file_path)
    for dynamic_indicator_name, dynamic_indicator_params in dynamic_indicators.items():
        dynamic_indicator_object = get_dynamic_indicator(dynamic_indicator_name)
        dynamic_indicator_object.process(params)
        params_processor = dynamic_indicator_object.create_params_processor(params)
        fname = dynamic_indicator_object.create_file_name_process(params_processor)
        filename = os.path.join(path, fname + ".png")
        assert os.path.exists(filename) or not params_processor.get_param("execute")
