import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.dynamic_indicators.poincare_maps import (
    PoincareMapFunction,
    get_poincare_grid_method,
)


@pytest.fixture
def lorentz_system():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    var = DiffVariable("x")

    def lorentz_function(variable: DiffVariable):
        x = variable.values[:, 1]
        y = variable.values[:, 2]
        z = variable.values[:, 3]
        return np.concatenate((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

    return DiffSystem(var, lorentz_function)


@pytest.fixture
def poincare_map():
    def function(states):
        return states[0]

    return function


@pytest.mark.parametrize("method", ["PoincareSectionInterpolate", "PoincareSectionOdeTimeRange"])
@pytest.mark.parametrize(
    "t_max, expected_t_roots, expected_values",
    [
        (10, np.array([0.51298646]), np.array([[5.21529486e-11, -8.82950611e00, 3.13821647e01]])),
        (
            15,
            np.array([0.51298646, 14.17828829, 14.95354952]),
            np.array(
                [
                    [-3.21422888e-11, -8.82950611e00, 3.13821647e01],
                    [-7.69295738e-12, 1.66700449e00, 2.17284880e01],
                    [1.27770017e-11, -1.95144391e00, 2.24303988e01],
                ]
            ),
        ),
    ],
)
def test_poincare_section_grid(
    method: str,
    t_max: float,
    expected_t_roots: np.ndarray,
    expected_values: np.ndarray,
    lorentz_system: DiffSystem,
    poincare_map: PoincareMapFunction,
):
    x0 = np.array([1.0, 1.0, 1.0])
    t_span = [0, t_max]
    n_points = t_max * 100
    poincare_method = get_poincare_grid_method(method)
    result_t_roots, result_values = poincare_method(
        diff_system=lorentz_system,
        poincare_map=poincare_map,
        solver_method="solve_ivp",
        t_span=t_span,
        x0=x0,
        n_points=n_points,
    )

    assert result_t_roots == pytest.approx(expected_t_roots, 1e-4)
    assert (np.abs(result_values - expected_values) < 1e-3).all()


def test_get_poincare_grid_method_error():
    with pytest.raises(ValueError):
        _ = get_poincare_grid_method("method_error")
