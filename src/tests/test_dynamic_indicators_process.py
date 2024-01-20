import json
import os

import pytest

from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_utils import (
    DynamicIndicatorNotExist,
    FtleElementWise,
    FtleGrid,
    FtleVariationalEquations,
    LagrangianDescriptor,
    PoincareSections,
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
        (FtleVariationalEquations.name_dynamic_indicator, FtleVariationalEquations),
        (LagrangianDescriptor.name_dynamic_indicator, LagrangianDescriptor),
        (PoincareSections.name_dynamic_indicator, PoincareSections),
    ],
)
def test_get_dynamic_indicator(di_method, expected):
    method_object = get_dynamic_indicator(di_method)
    assert isinstance(method_object, expected)


def test_get_dynamic_indicator_error():
    with pytest.raises(DynamicIndicatorNotExist):
        _ = get_dynamic_indicator("fail_method")


@pytest.mark.parametrize(
    "dynamic_indicator_object",
    [
        (FtleElementWise()),
        (FtleGrid()),
        (FtleVariationalEquations()),
        (LagrangianDescriptor()),
        (PoincareSections()),
    ],
)
def test_main_process(dynamic_indicator_object, config_main_test):
    with open(config_main_test, "r") as file_json:
        params = json.load(file_json)
    params_processor = dynamic_indicator_object.create_params_processor(params)
    fname = dynamic_indicator_object.create_file_name_process(params_processor)
    path = params_processor.get_param("path")
    filename = os.path.join(path, fname + ".png")
    if os.path.exists(filename):
        os.remove(filename)
    dynamic_indicator_object.process(params)
    assert os.path.exists(filename) or not params_processor.get_param("execute")
