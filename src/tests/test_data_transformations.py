from typing import List

import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.data_transformations import (
    Projection,
    project_grid_data,
)


def get_grid_points(n_var: int) -> List[np.ndarray]:
    """
    Función que dado un número de variables crea una lista de n_var elementos de
    mallas de puntos creados a través de numpy.meshgrid.

    Parameters
    ----------
    n_var: int
        Número de variables

    Returns
    -------
    grid_points: List[np.ndarray]
        Lista con la malla de puntos para cada variable de dimensión
        (n_var,)*2
    """
    x_min_grid = [0] * n_var
    x_max_grid = [1] * n_var
    nx_grid = 2
    grid_points = np.meshgrid(
        *[np.linspace(x_min_grid[i], x_max_grid[i], nx_grid) for i in range(n_var)]
    )
    return grid_points


def get_projection(n_var: int) -> Projection:
    """
    Función que dado un número de variables crea una proyección sobre l aúltima variable
    que calcula la media del resto.

    Parameters
    ----------
    n_var: int
        Número de variables

    Returns
    -------
    projection: Projection
        Proyección sobre la última variable que calcula la media del resto de dimensiones.
    """
    index_variables = list(range(n_var - 1))

    def function_projection(*variables: np.ndarray) -> np.ndarray:
        return np.mean(variables, axis=0)

    return Projection(index_variables, function_projection)


@pytest.mark.parametrize("n_var", [2, 5, 10])
def test_project_grid_data(n_var: int):
    grid_points = get_grid_points(n_var)
    projection = get_projection(n_var)
    projection_config = {n_var - 1: projection}
    proj_grid_points = project_grid_data(grid_points, projection_config)
    expected = np.zeros(proj_grid_points[-1].shape)
    for i in range(n_var - 1):
        expected += proj_grid_points[i]
    expected /= n_var - 1
    assert (np.abs(proj_grid_points[-1] - expected) < 1e-7).all()


@pytest.mark.parametrize("n_var", [3, 5, 10, 100])
def test_project_grid_data_array(n_var: int):
    variables = np.arange(n_var).astype(float)
    projection = get_projection(n_var)
    projection_config = {n_var - 1: projection}
    proj_variables = project_grid_data(variables, projection_config)
    expected = np.mean(proj_variables[:-1])
    assert (np.abs(proj_variables[-1] - expected) < 1e-7).all()
