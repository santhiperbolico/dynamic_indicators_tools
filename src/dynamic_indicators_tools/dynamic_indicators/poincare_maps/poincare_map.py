from typing import Callable, Union

import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import FlowMap
from dynamic_indicators_tools.dynamic_indicators.poincare_maps.poincare_utils import (
    PoincareMapFunction,
    roots_function,
)

PoincareMap = Callable[[np.ndarray], float]


def poincare_map_restriction_generator(
    poincare_map: PoincareMap, flow: FlowMap, x0: np.ndarray
) -> PoincareMapFunction:
    """
    Generador de la función flujo compuesta de g que representa el valor
    de la función de la sección de poincare de la imagen del flujo para t=t0 + tau
    de un sistema diferencial con condiciones iniciales (t0, x0).

    Parameters
    ----------
    poincare_map: PoincareMap
        Función de la sección de Poincaré
    flow: FlowMap
        Flujo del sistema diferencial
    x0: np.ndarray
        Valor de la condición inicial del sistema diferencial para flow.t0.

    Returns
    -------
    poincare_map_restriction: PoincareMapFunction
        Función que representa el valor de la función de la sección de poincare de la imagen
        del flujo para t=t0 + tau de un sistema diferencial con condiciones iniciales (t0, x0).
    """

    def poincare_map_restriction(tau: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Función que representa el valor de la función de la sección de poincare de la imagen
        del flujo para t=t0 + tau de un sistema diferencial con condiciones iniciales (t0, x0).

        Parameters
        ----------
        tau: float
            Desplazamiento en el tiempo del flujo.

        Returns
        -------
        g_tau: float
            Valor de la función de la sección de poincare de la imagen del flujo para t=t0 + tau.

        """
        if isinstance(tau, (float, int)):
            xi = flow(flow.t0 + tau, x0).T
            return poincare_map(xi)
        t = flow.t0 + tau.max()
        _ = flow.diff_system.solve_function(
            solver_method=flow.solver_method,
            t_span=[flow.t0, t],
            x0=x0,
            args=flow.args_func,
            params_solver=flow.params_solver,
        )
        xi = flow.diff_system.variable.solution(flow.t0 + tau)
        return poincare_map(xi)

    return poincare_map_restriction


def poincare_map_function(
    poincare_map: PoincareMap,
    flow: FlowMap,
    x0: np.ndarray,
    tau_max: Union[float, int],
    n_iter: int = 1,
    n_points: int = 10,
):
    poincare_map_restriction = poincare_map_restriction_generator(poincare_map, flow, x0)
    t_roots = roots_function(poincare_map_restriction, t_span=[0, tau_max], n_points=n_points)
    t_roots = t_roots + flow.t0
    if t_roots.size < n_iter:
        raise ValueError(
            "No se han encontrado suficientes raices en la sección de poincaré, "
            "pruebe a ampliar el t_span."
        )
    t_i = t_roots[n_iter - 1]
    return flow(t_i, x0)
