from typing import Callable, Tuple, Union

import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DiffVariable,
    FlowMap,
)
from dynamic_indicators_tools.dynamic_indicators.finite_time_lyapunov_exponents.ftle_utils import (
    diff_flow_grid,
    ftl_variational_equations,
    ftle_element_wise,
    ftle_fun,
    ftle_grid,
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


def get_system_variational_equations(dimension: int):
    x = DiffVariable("x")

    def fun_system(v: DiffVariable) -> np.ndarray:
        values = v.values[0, 1:]
        f_values = values
        f_values[:dimension] = f_values[:dimension] * np.power(-1, np.arange(dimension))
        f_values[dimension:] = f_values[dimension:] * np.power(
            -1, np.arange(dimension**2) // dimension
        )
        return f_values

    def solution(t: TimeType, x0: np.ndarray) -> np.ndarray:
        sol = np.zeros((1, dimension + dimension**2))
        diag_jac = np.diag(np.arange(dimension**2).reshape((dimension, dimension)))
        if isinstance(t, np.ndarray):
            sol = np.zeros((t.size, dimension))
        for i in range(dimension):
            p = (-1) ** i
            sol[:, i] = x0[i] * np.exp(p * t)
            j = diag_jac[i] + dimension
            sol[:, j] = np.exp(p * t)
        return sol

    dx = DiffSystem(x, fun_system)
    return dx, solution


@pytest.mark.parametrize("dimension, nx_grid", [(2, 10), (4, 4), (6, 2)])
def test_diff_flow(dimension: int, nx_grid: int):
    """
    Test que comprueba la funcionalidad del método diff_flow_grid que
    calcula una lista de elementos con las mallas de las derivadas
    parciales del flujo. Se utlizan como valor esperado
    las derivadas parciales para el sistema:
        dx_i = (-1)^(i) * x, i=0,.., dimension-1
    Cuya solución es x_i(t) = x0_i * exp(((-1)^i)* t)

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    nx_grid: int
        Número de puntos por cada dimensión de la variable del sistema.
    """
    t = 1
    n_jobs = 1

    dx, _ = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * dimension)
    x_max_grid = np.array([10] * dimension)
    grid_points, diff_f = diff_flow_grid(
        flow=flow_x, t=t, x0_min=x_min_grid, x0_max=x_max_grid, n_xgrid=nx_grid, n_jobs=n_jobs
    )
    for i in range(len(diff_f)):
        dfxi = diff_f[i]
        gradient = np.zeros(dimension)
        gradient[i] = np.exp((-1) ** i * t)
        expected_val = gradient.reshape((1, 1, -1)).repeat(nx_grid, axis=0).repeat(nx_grid, axis=1)
        error_max = np.abs(expected_val - dfxi).max()
        print(error_max)
        assert error_max < 5e-4


@pytest.mark.parametrize("dimension, nx_grid", [(2, 10), (4, 4), (6, 2)])
def test_ftle_grid(dimension: int, nx_grid: int):
    """
    Test que comprueba la funcionalidad del método ftle_grid que
    calcula una malla de los exponentes de Lyapunov en el tiempo t=1.
    Se utlizan como valor esperado una malla de unos ya que son los
    FTLE esperados para el problema:
        dx_i = (-1)^(i) * x, i=0,.., dimension-1
    Cuya solución es x_i(t) = x0_i * exp(((-1)^i)* t)

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    nx_grid: int
        Número de puntos por cada dimensión de la variable del sistema.
    """
    t = 1
    n_jobs = 1

    dx, _ = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * dimension)
    x_max_grid = np.array([10] * dimension)
    grid_points, ftle_grid_points = ftle_grid(flow_x, t, x_min_grid, x_max_grid, nx_grid, n_jobs)
    expected_value = np.ones(ftle_grid_points.shape)
    assert expected_value == pytest.approx(ftle_grid_points, 1e-4)
    assert len(grid_points) == dimension


@pytest.mark.parametrize(
    "x0, h_steps",
    [
        (np.array([0, 0]), np.array([0.01, 0.1])),
        (np.array([1, 5]), np.array([0.005, 0.001])),
    ],
)
def test_ftle_fun(x0: np.ndarray, h_steps: np.ndarray):
    t = 30
    dx, _ = get_diff_system()
    flow_x = FlowMap(dx, 0)
    result = ftle_fun(flow_x, t, x0, h_steps, params_t_close={})
    assert result == pytest.approx(1, 1e-3)


@pytest.mark.parametrize(
    "dimension, nx_grid, n_jobs, include_h",
    [(2, np.array([10, 7]), 4, True), (4, np.array([4, 3, 3, 2]), 1, False), (6, 2, 1, False)],
)
def test_ftle_element_wise(
    dimension: int, nx_grid: Union[int, np.ndarray], n_jobs: int, include_h: bool
):
    """
    Test que comprueba la funcionalidad del método ftle_grid que
    calcula una malla de los exponentes de Lyapunov en el tiempo t=1.
    Se utlizan como valor esperado una malla de unos ya que son los
    FTLE esperados para el problema:
        dx_i = (-1)^(i) * x, i=0,.., dimension-1
    Cuya solución es x_i(t) = x0_i * exp(((-1)^i)* t)

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    nx_grid: nx_grid: Union[int, np.ndarray]
        Número de puntos por cada dimensión de la variable del sistema.
    n_jobs: int
        Número máximo de hilos.
    include_h: bool
        Indica si queremos pasar un vector h_steps a la función ftle_element_wise
    """
    t = 1

    dx, _ = get_diff_system()
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * dimension)
    x_max_grid = np.array([10] * dimension)
    h_steps = None
    if include_h:
        h_steps = 1 / np.random.randint(10, 101, dimension)
    result = ftle_element_wise(flow_x, t, x_min_grid, x_max_grid, nx_grid, h_steps, n_jobs=n_jobs)
    expected_value = np.ones(result.ftle_grid.shape)
    expected_size = nx_grid**dimension
    if isinstance(nx_grid, np.ndarray):
        expected_size = nx_grid.prod()
    assert expected_value == pytest.approx(result.ftle_grid, 1e-4)
    assert len(result.grid_points) == dimension
    assert result.grid_points[0].size == expected_size


@pytest.mark.parametrize("dimension, t1", [(3, 10), (5, 5), (10, 3), (20, 1)])
def test_variational_equation_jacobian(dimension, t1):
    dx, solution = get_system_variational_equations(dimension)
    x0 = np.concatenate((np.ones(dimension), np.eye(dimension).reshape(dimension**2)))
    t, x = dx.solve_function("solve_ivp", [0, t1], x0)
    expected = solution(t1, x0)
    jac = x[-1, dimension:].reshape((dimension, dimension))
    expected = expected[0, dimension:].reshape((dimension, dimension))
    relative_error = np.abs(expected - jac) / (np.abs(expected) + 1e-8)
    assert (relative_error < 5e-2).all()


@pytest.mark.parametrize("dimension, nx_grid", [(2, 10), (4, 4), (5, 2)])
def test_ftl_variational_equations(dimension: int, nx_grid: int):
    """
    Test del cálculo de ftle con las ecuaciones variacionales.
    El máximo de dimensión posible es de 5, ya que el meshgrid no soporta
    dimensiones mayores a 32.

    Parameters
    ----------
    dimension: int
        Dimensión de la variable.
    nx_grid: int
        Número de puntos por cada dimensión de la variable del sistema.
    """
    t = 1
    n_jobs = 1

    dx, solution = get_system_variational_equations(dimension)
    flow_x = FlowMap(dx, 0)
    x_min_grid = np.array([0] * dimension)
    x_max_grid = np.array([10] * dimension)
    grid_points, ftle_grid_points = ftl_variational_equations(
        flow_x, t, x_min_grid, x_max_grid, nx_grid, n_jobs
    )
    expected_value = np.ones(ftle_grid_points.shape)
    assert expected_value == pytest.approx(ftle_grid_points, 1e-4)
    assert len(grid_points) == dimension
