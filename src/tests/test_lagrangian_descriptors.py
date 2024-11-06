from typing import Callable, Tuple, Union

import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.dynamic_indicators.lagrangian_descriptors.ld_integrators import (
    ParamsIntegratorLD,
    integrate_func_diff_system,
)
from dynamic_indicators_tools.dynamic_indicators.lagrangian_descriptors.ld_utils import ParamsLD
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
def test_integrate_func_diff_system_integrate(
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
    ld_method = "integrate"
    dx, _ = get_diff_system()
    ld_params = ParamsIntegratorLD(
        diff_system=dx,
        t0=0,
        t=2,
        x=x0,
        tau=tau,
        params_solver={"params_integrator": {"method": method}},
    )
    mean_velocity = integrate_func_diff_system(ld_method, ld_params)
    assert mean_velocity == pytest.approx(expected, 1e-2)


@pytest.mark.parametrize(
    "x0, tau, expected",
    [
        (np.array([0, 1]), 2, (1 - np.exp(-4)) / 4),
        (np.array([4, 0]), (0, 1), 4 * (np.exp(3) - np.exp(2)) / 1),
        (np.array([0, 2]), (1, 3), 2 * (np.exp(-1) - np.exp(-5)) / 4),
        (np.array([3, 0]), (2, 6), 3 * (np.exp(8) - np.exp(0)) / 8),
    ],
)
def test_integrate_func_diff_system_differential_equations(
    x0: np.ndarray, tau: Union[float, Tuple[float, float]], expected: float
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
    expected: float
        Valor esperado.
    """
    ld_method = "differential_equations"
    dx, _ = get_diff_system()
    ld_params = ParamsIntegratorLD(diff_system=dx, t0=0, t=2, x=x0, tau=tau)
    mean_velocity = integrate_func_diff_system(ld_method, ld_params)
    assert mean_velocity == pytest.approx(expected, 1e-2)


def test_integrate_func_diff_system_error():
    ld_method = "integrate"
    dx, _ = get_diff_system()
    ld_params = ParamsIntegratorLD(
        diff_system=dx,
        t0=0,
        t=2,
        x=np.array([0, 1]),
        tau=2,
        params_solver={"params_integrator": {"method": "error"}},
    )
    with pytest.raises(DoesntExistIntegrator):
        _ = integrate_func_diff_system(ld_method, ld_params)


def test_params_ld_get_params_integrators():
    dx, _ = get_diff_system()
    ld_params = ParamsLD(
        diff_system=dx,
        t0=0,
        t=2,
        tau=2,
        ld_method="integrate",
        params_solver={"params_integrator": {"method": "error"}},
        x0_min=np.array([0, 0, 0]),
        x0_max=np.array([1, 1, 1]),
        n_xgrid=10,
    )
    result = ld_params.get_params_integrators(x=np.array([1, 0, 0.5]))
    assert isinstance(result, ParamsIntegratorLD)
    assert result.diff_system == dx
    assert result.t0 == 0
    assert result.t == 2
    assert (result.x == np.array([1, 0, 0.5])).all()
    assert result.tau == 2
    assert result.params_solver == {"params_integrator": {"method": "error"}}
