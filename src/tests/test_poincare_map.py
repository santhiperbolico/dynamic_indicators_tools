import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DiffVariable,
    FlowMap,
)
from dynamic_indicators_tools.dynamic_indicators.poincare_maps.poincare_map import (
    diff_poincare_map_function,
    poincare_map_function,
    poincare_map_restriction_generator,
)
from dynamic_indicators_tools.dynamic_indicators.poincare_maps.poincare_utils import (
    PoincareMapFunction,
    get_poincare_grid_method,
    roots_function,
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
def lorenz_system_variational():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    var = DiffVariable("x")

    def lorenz_function(variable: DiffVariable) -> np.ndarray:
        """
        Función que construye el sistema de ecuaciones variacionales
        del sistema diferencial de Lorenz
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
        """
        x = variable.values[:, 1]
        y = variable.values[:, 2]
        z = variable.values[:, 3]

        g0 = variable.values[:, 4]
        g1 = variable.values[:, 5]
        g2 = variable.values[:, 6]
        g3 = variable.values[:, 7]
        g4 = variable.values[:, 8]
        g5 = variable.values[:, 9]
        g6 = variable.values[:, 10]
        g7 = variable.values[:, 11]
        g8 = variable.values[:, 12]

        fval0 = (sigma * (y - x)).reshape(-1, 1)
        fval1 = (x * (rho - z) - y).reshape(-1, 1)
        fval2 = (x * y - beta * z).reshape(-1, 1)

        fval3 = (-sigma * g0 + (rho - z) * g3 + y * g6).reshape(-1, 1)
        fval4 = (-sigma * g1 + (rho - z) * g4 + y * g7).reshape(-1, 1)
        fval5 = (-sigma * g2 + (rho - z) * g5 + y * g8).reshape(-1, 1)

        fval6 = (sigma * g0 + (-1) * g3 + x * g6).reshape(-1, 1)
        fval7 = (sigma * g1 + (-1) * g4 + x * g7).reshape(-1, 1)
        fval8 = (sigma * g2 + (-1) * g5 + x * g8).reshape(-1, 1)

        fval9 = (-x * g3 - beta * g6).reshape(-1, 1)
        fval10 = (-x * g4 - beta * g7).reshape(-1, 1)
        fval11 = (-x * g5 - beta * g8).reshape(-1, 1)

        return np.concatenate(
            (fval0, fval1, fval2, fval3, fval4, fval5, fval6, fval7, fval8, fval9, fval10, fval11),
            axis=1,
        )

    return DiffSystem(var, lorenz_function)


@pytest.fixture
def poincare_map():
    def function(states):
        return states[0]

    return function


@pytest.fixture
def gradient_poincare_map():
    def function(states):
        gradient = np.zeros(states.shape)
        gradient[0] = 1
        return gradient

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


@pytest.mark.parametrize(
    "t_span, t_array",
    [
        ([-2, 2], None),
        (None, np.array([-2, -1, 0, 1, 2])),
        (None, np.array([-2, -1.1, 0, 1.5, 2])),
    ],
)
def test_roots_function(t_span, t_array):
    def fun_to_null(t):
        return t**2 - 1

    result = roots_function(fun_to_null, t_span=t_span, n_points=8, t_array=t_array)
    assert result == pytest.approx(np.array([-1.0, 1.0]), abs=1e-7)


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


@pytest.mark.parametrize(
    "tau",
    [0.51298646, 14.17828829, 14.95354952, np.array([0.51298646, 14.17828829, 14.95354952])],
)
def test_poincare_map_restriction_generator(tau, poincare_map, lorenz_system):
    x0 = np.array([1.0, 1.0, 1.0])
    flow = FlowMap(lorenz_system, t0=0)
    flow.set_params_fun_solver(solver_method="solve_ivp")
    poincare_map_restriction = poincare_map_restriction_generator(poincare_map, flow, x0)
    expected = 0
    if isinstance(tau, np.ndarray):
        expected = np.zeros(tau.size)
    assert poincare_map_restriction(tau) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "n_iter, expected",
    [
        (1, np.array([[-3.21422888e-11, -8.82950611e00, 3.13821647e01]])),
        (2, np.array([[-7.69295738e-12, 1.66700449e00, 2.17284880e01]])),
        (3, np.array([[1.27770017e-11, -1.95144391e00, 2.24303988e01]])),
    ],
)
def test_poincare_map_function(n_iter, expected, poincare_map, lorenz_system):
    x0 = np.array([1.0, 1.0, 1.0])
    flow = FlowMap(lorenz_system, t0=0)
    flow.set_params_fun_solver(solver_method="solve_ivp")
    xi = poincare_map_function(poincare_map, flow, x0, 15, n_iter, n_points=750)
    assert xi == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize(
    "x0",
    [
        (np.array([1.0, 1.0, 1.0])),
        (np.array([0.8, 0.4, 0.3])),
        (np.array([0.7, 0.6, 0.8])),
    ],
)
def test_diff_poincare_map_function(
    x0, poincare_map, gradient_poincare_map, lorenz_system_variational
):
    n_iter = 1
    x0_var = np.concatenate((x0, np.eye(3).reshape(9)))
    flow = FlowMap(lorenz_system_variational, t0=0)
    flow.set_params_fun_solver(solver_method="solve_ivp")
    jac_projection = diff_poincare_map_function(
        poincare_map=poincare_map,
        gradient_poincare_map=gradient_poincare_map,
        flow=flow,
        x0=x0_var,
        tau_max=15,
        n_iter=n_iter,
        n_points=750,
    )
    assert jac_projection.shape == (3, 3)
    assert jac_projection[0, :] == pytest.approx(np.zeros(3), abs=1e-7)
