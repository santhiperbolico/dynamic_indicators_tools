from typing import Callable, Tuple, Union

import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.dynamic_indicators.lagrangian_descriptors.ld_utils import (
    integrate_func_diff_system,
)
from dynamic_indicators_tools.numercial_methods.integrators import DoesntExistIntegrator

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


@pytest.mark.parametrize(
    "x0, tau, method,expected",
    [
        (np.array([0, 1]), 2, "quad", (1 - np.exp(-4)) / 4),
        (np.array([4, 0]), (0, 1), "newton_cotes", 4 * (np.exp(3) - np.exp(2)) / 1),
        (np.array([0, 2]), (1, 3), "quadrature", 2 * (np.exp(-1) - np.exp(-5)) / 4),
        (np.array([3, 0]), (2, 6), "fixed_quad", 3 * (np.exp(8) - np.exp(0)) / 8),
    ],
)
def test_integrate_func_diff_system(
    x0: np.ndarray, tau: Union[float, Tuple[float, float]], method: str, expected: float
):
    """
    Test que comprueba la función que calcula la integral de la norma de la función f(t,x)
    del sistema diferencial utilizando diferentes métodos de integración.

    Parameters
    ----------
    x0: np.ndarray
        Valor de x que se fija en la integral sobre t.
    tau: Union[float, Tuple[float, float]]
        Valor utilizado para construir el intervalo de integración.
        Si es int o float el intervalo se calcula como [t-tau, t+tau].
        Si es una tupla de dos valores se calcula como [t-tau[0], t+tau[1]].
    method: str
        Método de integración utilizado.
    expected: float
        Valor esperado.
    """
    t = 2
    dx, _ = get_diff_system()
    mean_velocity, _ = integrate_func_diff_system(dx, t, x0, tau, method)
    assert mean_velocity == pytest.approx(expected, 1e-2)


def test_integrate_func_diff_system_error():
    dx, _ = get_diff_system()
    with pytest.raises(DoesntExistIntegrator):
        _, _ = integrate_func_diff_system(dx, 2, np.array([0, 1]), 1, "error")
