from typing import Sequence

import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable


def function_system(v: DiffVariable, b: float, c: float) -> np.ndarray:
    """
    Función que construye el sistema diferencia del péndulo no lineal.
        dx = y
        dy = -by - c*sin(x)

    Parameters
    ----------
    v: DiffVariable
        Variables del sistema diferencial.
    b: float
        Coeficiente de amortiguación
    c; float
        Coeficiente de la amplitud,

    Returns
    -------
    dv: np.ndarray
        Valores de f(v, t)
    """
    fval0 = v.values[:, 2].reshape(-1, 1)
    fval1 = (-b * v.values[:, 2] - c * np.sin(v.values[:, 1])).reshape(-1, 1)
    return np.concatenate((fval0, fval1), axis=1)


def extremals_functionals(
    x_min: np.ndarray, x_max: np.ndarray, b: float, c: float, n_points: int = 500
) -> Sequence[np.ndarray]:
    """
    Cálculo de las curvas extremales del sistema del péndulo.

    Parameters
    ----------
    x_min: np.ndarray
        Valores mínimos de x
    x_max: np.ndarray
        Valores máximos de x
    b: float
        Coeficiente de amortiguación
    c; float
        Coeficiente de la amplitud,
    n_points: int
        Número de puntos de las curvas extremales.

    Returns
    -------
    x_pc: np.ndarray
        Valores de las x de los puntos críticos
    y_pc: np.ndarray
        Valores de las y de los puntos críticos
    x_array: np.ndarray
        Valores de las x de la curva de lso extremales.
    y_dy: np.ndarray
        Valores de la y para las curvas extremales.
    """
    # Cálculo para las curvas que minimizan el funcional del descriptor lagrangiano.
    x_array = np.linspace(x_min[0], x_max[0], n_points)
    x_pc = np.arange(-np.pi, 3 * np.pi / 2, np.pi / 2)
    y_dy = -c * b * np.sin(x_array) / (1 + b**2)
    y_pc = -c * b * np.sin(x_pc) / (1 + b**2)
    return [x_pc, x_array, y_dy, y_pc]


def function_system_variational_equations(v: DiffVariable, b: float, c: float) -> np.ndarray:
    """
    Función que construye el sistema de ecuaciones variacionales del péndulo no lineal.
        dx = y
        dy = -by - c*sin(x)

    Parameters
    ----------
    v: DiffVariable
        Variables del sistema diferencial.
    b: float
        Coeficiente de amortiguación
    c; float
        Coeficiente de la amplitud,

    Returns
    -------
    dv: np.ndarray
        Valores de f(v, t)
    """
    fval0 = v.values[:, 2].reshape(-1, 1)
    fval1 = (-b * v.values[:, 2] - c * np.sin(v.values[:, 1])).reshape(-1, 1)
    fval2 = (-c * v.values[:, 5] * np.cos(v.values[:, 1])).reshape(-1, 1)
    fval3 = (-c * v.values[:, 6] * np.cos(v.values[:, 1])).reshape(-1, 1)
    fval4 = (v.values[:, 3] - b * v.values[:, 5]).reshape(-1, 1)
    fval5 = (v.values[:, 4] - b * v.values[:, 6]).reshape(-1, 1)
    return np.concatenate((fval0, fval1, fval2, fval3, fval4, fval5), axis=1)
