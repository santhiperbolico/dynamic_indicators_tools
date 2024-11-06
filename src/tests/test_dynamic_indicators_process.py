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
    main_process_di,
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


def test_main_process_di(config_main_test):
    with open(config_main_test, "r") as file_json:
        params = json.load(file_json)
    path = params.get("system_params").get("path")
    main_process_di(params)
    plot_tests = [
        "test_system_ftle_grid_t_10_nx_grid_10.png",
        "test_system_poincare_section_t_10_nx_grid_100.png",
        "test_system_ftle_variational_equations_t_10_nx_grid_10.png",
        "test_system_lagrangian_descriptors_differential_equations_t_10_nx_grid_10.png",
        "test_system_ftle_element_wise_t_10_nx_grid_10_h_0.0100.png",
    ]
    for plot in plot_tests:
        filename = os.path.join(path, plot)
        assert os.path.exists(filename)
        os.remove(filename)
