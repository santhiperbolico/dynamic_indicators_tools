from typing import Callable, Union

import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable, FlowMap
from dynamic_indicators_tools.dynamic_indicators.poincare_maps.poincare_utils import (
    PoincareMapFunction,
    roots_function,
)

PoincareMap = Callable[[np.ndarray], float]
GradientPoincareMap = Callable[[np.ndarray], np.ndarray]


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


def get_t_root_n_iter(
    poincare_map: PoincareMap,
    flow: FlowMap,
    x0: np.ndarray,
    tau_max: Union[float, int],
    n_iter: int = 1,
    n_points: int = 10,
) -> Union[float, int]:
    """
    Función que calcula el valor t_i donde la órbita del flujo de condición inicial x0 ha pasado
    n_iter veces por la sección de poincaré.

    Parameters
    ----------
    poincare_map: PoincareMap
        Función R^n->R que genera la sección de Poincaré.
    flow: FlowMap
        Flujo del sistema diferencial.
    x0: np.ndarray
        Condición inicial del sistema para fow.t0.
    tau_max: Union[float, int]
        Incremento del tiempo donde se van a buscar los valores de la función.
        El sistema diferencial se resolverá en un intervalo [flow.t0, flow.t0 + tau_max].
    n_iter: int, default 1
        Número de pasos por la sección de Poincaré.
    n_points: int, default 10
        Número de puntos para los que se van a buscar soluciones
        en el intervalo [flow.t0, flow.t0 + tau_max]

    Returns
    -------
    t_i: Union[float, int]
        Valor del tiempo dónde la órbita del flujo para x0 ha pasado n_iter veces por la sección
        de poincaré.
    """
    poincare_map_restriction = poincare_map_restriction_generator(poincare_map, flow, x0)
    t_roots = roots_function(poincare_map_restriction, t_span=[0, tau_max], n_points=n_points)
    t_roots = t_roots + flow.t0
    if t_roots.size < n_iter:
        raise ValueError(
            "No se han encontrado suficientes raices en la sección de poincaré, "
            "pruebe a ampliar el t_span."
        )
    t_i = t_roots[n_iter - 1]
    return t_i


def poincare_map_function(
    poincare_map: PoincareMap,
    flow: FlowMap,
    x0: np.ndarray,
    tau_max: Union[float, int],
    n_iter: int = 1,
    n_points: int = 10,
) -> np.ndarray:
    """
    Función que devuelve el valor del flujo para las n_iter pasos de la órbita del flujo
    en la sección de poincaré marcada por poincare_map.

    Parameters
    ----------
    poincare_map: PoincareMap
        Función R^n->R que genera la sección de Poincaré.
    flow: FlowMap
        Flujo del sistema diferencial.
    x0: np.ndarray
        Condición inicial del sistema para fow.t0.
    tau_max: Union[float, int]
        Incremento del tiempo donde se van a buscar los valores de la función.
        El sistema diferencial se resolverá en un intervalo [flow.t0, flow.t0 + tau_max].
    n_iter: int, default 1
        Número de pasos por la sección de Poincaré.
    n_points: int, default 10
        Número de puntos para los que se van a buscar soluciones
        en el intervalo [flow.t0, flow.t0 + tau_max]

    Returns
    -------
    xi: np.ndarray
        Punto en la sección de poincaré igual al valor de la órbita del flujo para x0, después
        de haber pasado n_iter veces por la sección.
    """
    t_i = get_t_root_n_iter(poincare_map, flow, x0, tau_max, n_iter, n_points)
    return flow(t_i, x0)


def get_dimension_from_variational_equations(x0: np.ndarray) -> int:
    """
    Función que extrae la dimensión de las variables de un sistema diferencial
    de una condición inicial correspondiente de su sistema diferencial de ecuaciones
    variacionales.

    Parameters
    ----------
    x0: np.ndarray
        Condición inicial de un sistema diferencial de ecuaciones variacionales.

    Returns
    -------
    n: int
        Dimensión del problema original, eliminando las ecuaciones variacionales.
    """
    m = x0.size
    n = (np.sqrt(1 + 4 * m) - 1) / 2
    if n % 1 > 0:
        raise ValueError(
            "La dimensión de x0 no se corresponde a la de una condición"
            " inicial de un sistema de ecuaciones variacionales."
        )
    return int(n)


def diff_poincare_map_function(
    poincare_map: PoincareMap,
    gradient_poincare_map: GradientPoincareMap,
    flow: FlowMap,
    x0: np.ndarray,
    tau_max: Union[float, int],
    n_iter: int = 1,
    n_points: int = 10,
) -> np.ndarray:
    """
    Función que devuelve el valor del jacobiano de la función poincare_map_function, la cual
    devuelve el valor del flujo para las n_iter pasos de la órbita del flujo en la sección de
    poincaré marcada por poincare_map.

    Parameters
    ----------
    poincare_map: PoincareMap
        Función R^n->R que genera la sección de Poincaré.
    gradient_poincare_map: GradientPoincareMap
        Función R^n->R^n del gradiente de poincare_map
    flow: FlowMap
        Flujo del sistema diferencial.
    x0: np.ndarray
        Condición inicial del sistema para fow.t0.
    tau_max: Union[float, int]
        Incremento del tiempo donde se van a buscar los valores de la función.
        El sistema diferencial se resolverá en un intervalo [flow.t0, flow.t0 + tau_max].
    n_iter: int, default 1
        Número de pasos por la sección de Poincaré.
    n_points: int, default 10
        Número de puntos para los que se van a buscar soluciones
        en el intervalo [flow.t0, flow.t0 + tau_max]

    Returns
    -------
    xi: np.ndarray
        Punto en la sección de poincaré igual al valor de la órbita del flujo para x0, después
        de haber pasado n_iter veces por la sección.
    """
    t_i = get_t_root_n_iter(poincare_map, flow, x0, tau_max, n_iter, n_points)
    values = flow(t_i, x0).T
    t_i_array = np.ones((values.shape[1], 1)) * t_i
    n_var = get_dimension_from_variational_equations(values)
    xi, jac_flow = values[:n_var], values[n_var:].reshape((n_var, n_var))
    df_var_xi = DiffVariable("xi", np.concatenate((t_i_array, values.T), axis=1))
    f_xi = flow.diff_system.function(df_var_xi).reshape(-1, 1)

    projection = np.dot(gradient_poincare_map(xi).T, jac_flow) / np.dot(
        gradient_poincare_map(xi).T, f_xi[:n_var]
    )
    projection = np.dot(f_xi[:n_var], projection)
    return jac_flow - projection
