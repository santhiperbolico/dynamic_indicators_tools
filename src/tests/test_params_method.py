import numpy as np
import pytest

from dynamic_indicators_tools.dynamic_indicators.params_methods import format_symbolic_number


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("-_pi", -np.pi),
        ("2_-_pi", 2 - np.pi),
        ("0.12_/_3.8", 0.12 / 3.8),
        ("e_^_-2", np.exp(-2)),
        ("e_*_pi", np.exp(1) * np.pi),
        ("tau_+_2", 2 * np.pi + 2),
        ("0.12_/_3.8", 0.12 / 3.8),
        (2.93, 2.93),
    ],
)
def test_format_symbolic_number(symbol, expected):
    result = format_symbolic_number(symbol)
    assert result == pytest.approx(expected, 1e-4)


def test_format_symbolic_number_error():
    with pytest.raises(RuntimeError):
        format_symbolic_number("fail_value")


@pytest.fixture
def config_main_test():
    return "tests/config_files/config_main_test_system.json"


#
# def test_get_main_params_system_params(config_main_test):
#     system, system_params, dynamic_indicators = get_main_params(config_main_test)
#
#     expected_system_params = {
#         "path": "tests/systems/test_plots/",
#         "system_name": "test_system_2",
#         "args_system": [2],
#         "t0": 0,
#         "t1": 10,
#         "x_min": np.array([-np.pi, -3.0, 0.0]),
#         "x_max": np.array([np.pi, 3.0, 1.0]),
#         "n_xgrid": 10,
#         "solver_method": "solve_ivp",
#         "n_jobs": 1,
#     }
#
#     assert expected_system_params["path"] == system_params["path"]
#     assert expected_system_params["system_name"] == system_params["system_name"]
#     assert expected_system_params["args_system"] == system_params["args_system"]
#     assert expected_system_params["t0"] == system_params["t0"]
#     assert (expected_system_params["x_min"] == system_params["x_min"]).all()
#     assert (expected_system_params["x_min"] == system_params["x_min"]).all()
#     assert expected_system_params["n_xgrid"] == system_params["n_xgrid"]
#     assert expected_system_params["solver_method"] == system_params["solver_method"]
#     assert expected_system_params["n_jobs"] == system_params["n_jobs"]
#
#
# def test_get_main_params_dynamic_params(config_main_test):
#     system, system_params, dynamic_indicators = get_main_params(config_main_test)
#
#     expected_dynamic = {
#         "ftle_element_wise": {
#             "execute": True,
#             "h_steps": 0.01,
#             "t_close": True,
#             "params_t_close": {
#                 "time_delta": 0.2,
#                 "dimensions_close": [True, False, False],
#                 "mod_solution": 2 * np.pi,
#             },
#         },
#         "ftle_grid": {"execute": True},
#         "lagrangian_descriptors": {
#             "execute": True,
#             "tau": 5,
#             "method_integrate": "fixed_quad",
#             "plot_orbits": True,
#         },
#     }
#     execute_ftle_element_wise = dynamic_indicators["ftle_element_wise"]["execute"]
#     h_steps_ftle_element_wise = dynamic_indicators["ftle_element_wise"]["h_steps"]
#     t_close_ftle_element_wise = dynamic_indicators["ftle_element_wise"]["t_close"]
#     time_delta = dynamic_indicators["ftle_element_wise"]["params_t_close"]["time_delta"]
#     dim_c = dynamic_indicators["ftle_element_wise"]["params_t_close"]["dimensions_close"]
#     mod_solution = dynamic_indicators["ftle_element_wise"]["params_t_close"]["mod_solution"]
#
#     execute_ftle_grid = dynamic_indicators["ftle_grid"]["execute"]
#     tau_lagrangian_descriptors = dynamic_indicators["lagrangian_descriptors"]["tau"]
#     execute_lagrangian_descriptors = dynamic_indicators["lagrangian_descriptors"]["execute"]
#     method_integrate = dynamic_indicators["lagrangian_descriptors"]["method_integrate"]
#     plot_orbits = dynamic_indicators["lagrangian_descriptors"]["plot_orbits"]
#
#     assert expected_dynamic["ftle_element_wise"]["execute"] == execute_ftle_element_wise
#     assert expected_dynamic["ftle_element_wise"]["h_steps"] == h_steps_ftle_element_wise
#     assert expected_dynamic["ftle_element_wise"]["t_close"] == t_close_ftle_element_wise
#     assert expected_dynamic["ftle_element_wise"]["params_t_close"]["time_delta"] == time_delta
#     assert expected_dynamic["ftle_element_wise"]["params_t_close"]["dimensions_close"] == dim_c
#    assert expected_dynamic["ftle_element_wise"]["params_t_close"]["mod_solution"] == mod_solution
#
#     assert expected_dynamic["ftle_grid"]["execute"] == execute_ftle_grid
#    assert expected_dynamic["lagrangian_descriptors"]["execute"] == execute_lagrangian_descriptors
#     assert expected_dynamic["lagrangian_descriptors"]["tau"] == tau_lagrangian_descriptors
#     assert expected_dynamic["lagrangian_descriptors"]["method_integrate"] == method_integrate
#     assert expected_dynamic["lagrangian_descriptors"]["plot_orbits"] == plot_orbits
