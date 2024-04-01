import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.dynamic_indicators.poincare_maps.poincare_utils import (
    PoincareMapFunction,
    get_poincare_grid_method,
)


@pytest.fixture
def lorenz_system():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    var = DiffVariable("x")

    def lorenz_function(variable: DiffVariable):
        x = variable.values[:, 1]
        y = variable.values[:, 2]
        z = variable.values[:, 3]
        return np.concatenate((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

    return DiffSystem(var, lorenz_function)


@pytest.fixture
def poincare_map():
    def function(states):
        return states[0]

    return function


@pytest.fixture
def lorenz_points_grid():
    x0_grid = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.1])]
    expected_values_roots = [
        np.array(
            [
                [-3.21200844e-11, -8.82950611e00, 3.13821647e01],
                [-7.66098296e-12, 1.66700449e00, 2.17284880e01],
                [1.27036159e-11, -1.95144391e00, 2.24303988e01],
            ]
        ),
        np.array(
            [
                [2.92743607e-11, -8.78467604e00, 3.13432023e01],
                [-1.01962883e-12, 4.61779124e00, 2.69790229e01],
            ]
        ),
    ]
    expected_t_roots = [
        np.array([0.51298646, 14.17828829, 14.95354952]),
        np.array([0.51380238, 14.31759899]),
    ]
    t_max = 15
    return t_max, x0_grid, expected_t_roots, expected_values_roots


def test_get_poincare_grid_method_error():
    with pytest.raises(ValueError):
        _ = get_poincare_grid_method("method_error")


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
def test_poincare_section_grid_get_poincare_points(
    method: str,
    t_max: float,
    expected_t_roots: np.ndarray,
    expected_values: np.ndarray,
    lorenz_system: DiffSystem,
    poincare_map: PoincareMapFunction,
):
    x0 = np.array([1.0, 1.0, 1.0])
    t_span = [0, t_max]
    n_points = int(t_max * 50)
    poincare_method = get_poincare_grid_method(method).get_poincare_points
    result_t_roots, result_values = poincare_method(
        diff_system=lorenz_system,
        poincare_map=poincare_map,
        solver_method="solve_ivp",
        t_span=t_span,
        x0=x0,
        n_points=n_points,
    )

    assert result_t_roots == pytest.approx(expected_t_roots, 1e-3)
    assert (np.abs(result_values - expected_values) < 5e-2).all()


@pytest.mark.parametrize("method", ["PoincareSectionInterpolate", "PoincareSectionOdeTimeRange"])
@pytest.mark.parametrize("x0_grid_is_array", [True, False])
def test_poincare_section_get_poincare_points_from_x0_grid(
    method: str,
    x0_grid_is_array: bool,
    lorenz_system: DiffSystem,
    poincare_map: PoincareMapFunction,
    lorenz_points_grid,
):
    t_max, x0_grid, expected_t_roots, expected_values_roots = lorenz_points_grid
    t_span = [0, t_max]
    n_points = int(t_max * 50)
    poincare_method = get_poincare_grid_method(method).get_poincare_points_from_x0_grid
    x0_grid_values = np.array(x0_grid) if x0_grid_is_array else x0_grid
    result_t_roots, result_values = poincare_method(
        diff_system=lorenz_system,
        poincare_map=poincare_map,
        solver_method="solve_ivp",
        t_span=t_span,
        x0_grid=x0_grid_values,
        n_points=n_points,
    )

    for i in range(len(x0_grid)):
        assert result_t_roots[i] == pytest.approx(expected_t_roots[i], 1e-2)
        assert (np.abs(result_values[i] - expected_values_roots[i]) < 1).all()
