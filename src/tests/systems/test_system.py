from typing import Sequence, Union

import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable


def function_system(v: DiffVariable, b: float) -> np.ndarray:
    values = v.values[:, 1:]
    dimension = values.shape[1]
    return values * np.power(-1, np.arange(dimension)) * b


def fun_system_variational_equations(v: DiffVariable, b: float) -> np.ndarray:
    dimension = 3
    values = v.values[0, 1:]
    f_values = values
    f_values[:dimension] = f_values[:dimension] * np.power(-1, np.arange(dimension)) * b
    f_values[dimension:] = f_values[dimension:] * np.power(
        -1, np.arange(dimension**2) // dimension
    )
    return f_values


def extremals_functionals(
    x_min: np.ndarray, x_max: np.ndarray, b: float, n_points: int = 500
) -> Sequence[np.ndarray]:
    # Cálculo para las curvas que minimizan el funcional del descriptor lagrangiano.
    x_array = np.linspace(x_min[0], x_max[0], n_points)
    x_pc = np.array([0])
    y_dy = x_array * 0 * b
    y_pc = np.array([0])
    return [x_pc, x_array, y_dy, y_pc]


def projection_generator():
    """
    Función que devuelve el valor de py para un valor h0 dado del hamiltoniano.
    """

    def projection(*variables: np.ndarray) -> np.ndarray:
        return np.sum(variables, axis=0) * 0

    return projection


def poincare_map_function(x: np.ndarray):
    return x[1] - x[0]


def poincare_initial_conditions_function(
    x_min: Union[Sequence[float], np.ndarray],
    x_max: Union[Sequence[float], np.ndarray],
    n_points: int,
):
    if isinstance(x_min, list):
        x_min = np.array(x_min)
    if isinstance(x_max, list):
        x_max = np.array(x_max)
    steps = np.linspace(0, 1, n_points)
    return [(x_max - x_min) * step + x_min for step in steps]
