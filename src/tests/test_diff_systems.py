from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pytest
from scipy.interpolate import interp1d

from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DiffVariable,
    DoesntCoincideDimension,
    DoesntExistSolution,
    FlowMap,
)
from dynamic_indicators_tools.differentials_systems.solver_integrators import (
    DoesntExisteSolverIntegerMethod,
)

TimeType = Union[int, float, np.ndarray]
SolutionSystem = Callable[[TimeType, np.ndarray], np.ndarray]


def get_diff_system() -> Tuple[DiffSystem, SolutionSystem]:
    """
    Funicón que genera DiffSysten y su solución analítica de:
        dx_i = (-1)^(i) * x, i=0,.., dimension-1
    Cuya solución es x_i(t) = x0_i * exp(((-1)^i)* t)

    Returns
    -------
    dx: DiffSystem
        Sistema diferencial generado
    solution: SolutionSystem
        Función con la solución el sistema
    """
    x = DiffVariable("x")

    def fun_system(v: DiffVariable) -> np.ndarray:
        values = v.values[:, 1:]
        dimension = values.shape[1]
        return values * np.power(-1, np.arange(dimension))

    def solution(t: TimeType, x0: np.ndarray) -> np.ndarray:
        dimension = x0.size
        sol = np.zeros((1, dimension))
        if isinstance(t, np.ndarray):
            sol = np.zeros((t.size, dimension))
        for i in range(dimension):
            p = (-1) ** i
            sol[:, i] = x0[i] * np.exp(p * t)
        return sol

    dx = DiffSystem(x, fun_system)
    return dx, solution


def test_diff_variables_set_values():
    """
    Tets que comprueba el método set_values de diff vaiables
    """
    x = DiffVariable("x")
    t_values = np.array([0, 1, 2, 3])
    x_values = np.array([[0, 1], [2, 3], [9, 4], [3, 9]])
    x.set_values(t_values, x_values)
    assert (t_values == x.values[:, 0]).all()
    assert (x_values == x.values[:, 1:]).all()


def test_diff_variables_solution_error():
    """
    Tets que comprueba el error del método set_values de diff vaiables
    """
    x = DiffVariable("x")
    t_values = np.array([0, 1, 2, 3])
    with pytest.raises(DoesntExistSolution):
        _ = x(t_values)


def test_diff_variables_set_values_error_dimension():
    """
    Tets que comprueba el error del método set_values de diff vaiables
    """
    x = DiffVariable("x")
    t_values = np.array([0, 1, 2, 3])
    x_values = np.array([[0, 1], [2, 3]])
    with pytest.raises(DoesntCoincideDimension):
        x.set_values(t_values, x_values)


def test_diff_variables_solution():
    """
    Tets que comprueba el método set_slolution de diff vaiables
    """
    x = DiffVariable("x")
    t_values = np.array([0, 1, 2, 3])
    x_values = np.array([[0, 1], [2, 3], [9, 4], [3, 9]])

    def x_solution(t: np.ndarray) -> np.ndarray:
        return interp1d(t_values, x_values.T, kind="slinear")(t).T

    x.set_solution(x_solution)
    assert (x(t_values) == x_values).all()
    assert (x(0.5) == np.array([1.0, 2.0])).all()


@pytest.mark.parametrize(
    "solver_method, dimension", [("solve_ivp", 2), ("solve_ivp", 4), ("odeint", 2), ("odeint", 4)]
)
def test_diff_solution_system(solver_method, dimension):
    """
    Tets que comprueba el método solver de diff systema
    """
    dx, expected_solution = get_diff_system()
    x0 = np.random.randint(-10, 10, size=dimension)
    t_array, x_array = dx.solve_function(solver_method=solver_method, t_span=[0, 1], x0=x0)
    assert expected_solution(t_array, x0) == pytest.approx(x_array, 1e-3)


def test_diff_solution_system_error():
    """
    Tets que comprueba el error del método solver de diff systema
    """
    dx, _ = get_diff_system()
    x0 = np.array([1, -1])
    with pytest.raises(DoesntExisteSolverIntegerMethod):
        _, _ = dx.solve_function(solver_method="solver_fail", t_span=[0, 1], x0=x0)


@pytest.mark.parametrize("dimension", [2, 4, 6])
def test_flow_call(dimension: int):
    """
    Test que comrueba que el método __call_ de Flow Map
    devuelve el valor esperado.

    Parameters
    ----------
    dimension: int
        Dimensión del problema
    """
    dx, expected_solution = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x0 = np.random.randint(-10, 10, size=(4, dimension))
    result = flow_x(1, x0)
    expected_values = np.zeros(x0.shape)
    for i in range(x0.shape[0]):
        expected_values[i, :] = expected_solution(1, x0[i, :])
    assert expected_values == pytest.approx(result, 1e-3)


@pytest.mark.parametrize(
    "dimension, nx_grid", [(2, np.array([4, 2])), (4, np.array([4, 2, 2, 3])), (6, 2)]
)
def test_flow_grid(dimension: int, nx_grid: Union[int, Sequence[int]]):
    """
    Test que comprueba la funcionalidad del método flow_grid que
    calcula una malla del flujo de un sistema dinámico.

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    """
    dx, expected_solution = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * dimension)
    x_max_grid = np.array([10] * dimension)
    grid_points, result = flow_x.flow_grid(1, x_min_grid, x_max_grid, nx_grid, n_jobs=1)
    it = np.nditer(grid_points[0], flags=["multi_index"])

    expected_val = np.zeros(result.shape)
    expected_size = nx_grid**dimension
    if isinstance(nx_grid, np.ndarray):
        expected_size = nx_grid.prod()

    while not it.finished:
        x0 = np.array([gp[it.multi_index] for gp in grid_points])
        expected_val[it.multi_index] = expected_solution(1, x0)
        _ = it.iternext()
    assert expected_val == pytest.approx(result, 1e-3)
    assert len(grid_points) == dimension
    assert grid_points[0].size == expected_size


def test_flow_grid_error_dimension_grid():
    """
    Test que comprueba la funcionalidad del método flow_grid que
    calcula una malla del flujo de un sistema dinámico. En este caso
    comprueba el error DoesntCoincideDimension cuando los limites
    de la malla no coinciden en dimensión.

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    """
    dx, expected_solution = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * 2)
    x_max_grid = np.array([10] * 3)
    nx_grid = 2
    with pytest.raises(DoesntCoincideDimension):
        _, _ = flow_x.flow_grid(1, x_min_grid, x_max_grid, nx_grid, n_jobs=1)


def test_flow_grid_error_dimension_nx_grid():
    """
    Test que comprueba la funcionalidad del método flow_grid que
    calcula una malla del flujo de un sistema dinámico. En este caso
    comprueba el error DoesntCoincideDimension cuando nx_grid es un
    array y no coincide con el número de variables

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    """
    dx, expected_solution = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * 3)
    x_max_grid = np.array([10] * 3)
    nx_grid = np.array([2, 2])
    with pytest.raises(DoesntCoincideDimension):
        _, _ = flow_x.flow_grid(1, x_min_grid, x_max_grid, nx_grid, n_jobs=1)


def test_flow_time_close():
    """
    Test que comprueba la funcionalidad de flow_time_close
    que busca el valro de la solución más cercano a las condiciones
    iniciales.
    """
    x = DiffVariable("x")

    def pendulum_fun(v: DiffVariable) -> np.ndarray:
        fval0 = v.values[:, 2].reshape(-1, 1)
        fval1 = (-np.sin(v.values[:, 1])).reshape(-1, 1)
        return np.concatenate((fval0, fval1), axis=1)

    flow_map = FlowMap(diff_system=DiffSystem(x, pendulum_fun), t0=0)
    flow_map.set_params_fun_solver(solver_method="solve_ivp")
    t = 10
    x0 = np.array([-np.pi, -3])
    result0 = flow_map.get_time_close(t, x0, 0.2, [True, False], np.array([2 * np.pi]))
    expected0 = 11.2865731
    assert expected0 == pytest.approx(result0, 1e-4)
