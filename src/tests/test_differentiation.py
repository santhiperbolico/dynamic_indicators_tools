from typing import Callable, Tuple, Union

import numpy as np
import pytest

from dynamic_indicators_tools.numercial_methods.differentiation import (
    diff_num_function,
    diff_partials_grid,
    jacobian_matrix,
)

FunctionType = Callable[[np.ndarray], np.ndarray]


def get_diff_function() -> Tuple[FunctionType, FunctionType]:
    """
    Funicón que devuelva la función:
        f(x) = [cos(x), sin(x)]
    y su derivada:
        f'(x) = [-sin(x), cos(x)]

    Returns
    -------
    function: FunctionType
        Función f(x)
    dfunction: FunctionType
        Derivada de la función f(x)
    """

    def function(x: Union[float, int, np.ndarray]) -> np.ndarray:
        fval0 = np.cos(x)
        fval1 = np.sin(x)
        if isinstance(x, np.ndarray):
            if len(x.shape) > 1:
                fval0 = fval0[:, 0]
                fval1 = fval1[:, 0]
            return np.array([fval0, fval1]).T
        return np.array([fval0, fval1])

    def dfunction(x: np.ndarray, partial_i: int) -> np.ndarray:
        fval0 = -np.sin(x)
        fval1 = np.cos(x)
        if isinstance(x, np.ndarray):
            if len(x.shape) > 1:
                fval0 = fval0[:, 0]
                fval1 = fval1[:, 0]
            df = np.array([fval0, fval1]).T
            if partial_i > 0:
                df = df * 0
            return df
        return np.array([fval0, fval1])

    return function, dfunction


@pytest.mark.parametrize(
    "n_degree,x_point, h_step, partial_i",
    [
        (
            1,
            np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 0]),
            np.array([0.01]),
            0,
        ),
        (
            1,
            np.array([[0, np.pi / 4], [np.pi / 2, 3 * np.pi / 4], [np.pi, 0]]),
            np.array([0.01, 0.03]),
            0,
        ),
        (
            1,
            np.array([[0, np.pi / 4], [np.pi / 2, 3 * np.pi / 4], [np.pi, 0]]),
            np.array([0.01, 0.03]),
            1,
        ),
        (
            2,
            np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 0]),
            np.array([0.01]),
            0,
        ),
        (
            2,
            np.array([[0, np.pi / 4], [np.pi / 2, 3 * np.pi / 4], [np.pi, 0]]),
            np.array([0.01, 0.03]),
            0,
        ),
        (
            2,
            np.array([[0, np.pi / 4], [np.pi / 2, 3 * np.pi / 4], [np.pi, 0]]),
            np.array([0.01, 0.03]),
            1,
        ),
    ],
)
def test_diff_num_function(n_degree: int, x_point: np.ndarray, h_step: np.ndarray, partial_i: int):
    """
    Test que comprueba la funcionalidad de las derivadas numéricas.

    Parameters
    ----------
    n_degree: int
        Grado del error de la derivada numérica
    x_point: np.ndarray
        Array con los puntos de test.
    h_step: np.ndarray
        Array con los pasos de la derivada
    partial_i: int
        Índice de la derivada parcial a calcular.
    """
    function, dfunction = get_diff_function()
    result = diff_num_function(function, x_point, h_step, partial_i=partial_i, n_degree=n_degree)
    expected = dfunction(x_point, partial_i)
    assert (np.abs(expected - result) < h_step[partial_i] ** n_degree).all()


@pytest.mark.parametrize(
    "x_point, h_step",
    [
        (
            np.array([[0, np.pi / 4], [np.pi / 2, 3 * np.pi / 4], [3 * np.pi / 4, np.pi]]),
            np.array([0.01, 0.03]),
        )
    ],
)
def test_jacobian_function(x_point: np.ndarray, h_step: np.ndarray):
    """
    Test que comprueba la funcionalidad de las derivadas numéricas.

    Parameters
    ----------
    x_point: np.ndarray
        Array con los puntos de test.
    h_step: np.ndarray
        Array con los pasos de la derivada
    """
    function, dfunction = get_diff_function()
    result = jacobian_matrix(function, x_point, h_step, n_degree=2)
    x_dim = x_point.shape[1]
    for k in range(x_point.shape[0]):
        expected_jac = np.zeros((x_dim, x_dim))
        for partial_i in range(x_dim):
            expected_jac[partial_i, :] = dfunction(x_point[[k], :], partial_i)
        assert (np.abs(expected_jac - result[k]) < np.max(h_step) ** 2).all()


@pytest.mark.parametrize("nx_grid, edge_remove", [(20, False), (10, True)])
def test_diff_partials_grid(nx_grid: int, edge_remove: bool):
    """
    Test que comprueba que sacamos la malla d ederivadas parciales
    dada dicha malla de puntos y sus valores con diff_partials_grid
    Parameters
    ----------
    nx_grid: int
        Número de puntos de la malla por dimensión
    edge_remove: bool
        Indica si se tiene en cuenta los bordes de la malla.
    """
    function, dfunction = get_diff_function()
    x0 = np.linspace(-np.pi, 0, nx_grid)
    y0 = np.linspace(0, np.pi, nx_grid)
    grid_points = np.meshgrid(x0, y0)
    function_values = np.zeros(grid_points[0].shape + (2,))
    dx_expected = np.zeros(grid_points[0].shape + (2,))
    dy_expected = np.zeros(grid_points[0].shape + (2,))
    it = np.nditer(grid_points[0], flags=["multi_index"])
    while not it.finished:
        x0 = np.array([gp[it.multi_index] for gp in grid_points]).reshape(1, 2)
        function_values[it.multi_index] = function(x0)
        dx_expected[it.multi_index] = dfunction(x0, 0)
        dy_expected[it.multi_index] = dfunction(x0, 1)
        _ = it.iternext()
    h_steps = np.array([np.pi, np.pi]) / (nx_grid - 1)
    diff_partials = diff_partials_grid(function_values, 2, h_steps, edge_remove)
    mask = (np.s_[:],) * 2
    error = h_steps
    if edge_remove:
        mask = (np.s_[1:-1],) * 2
        error = error**2
    assert np.abs(diff_partials[0] - dx_expected[mask]).max() < error[0]
    assert np.abs(diff_partials[1] - dy_expected[mask]).max() < error[1]
