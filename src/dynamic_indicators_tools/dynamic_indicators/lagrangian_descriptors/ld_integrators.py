from typing import Any, Dict, List, Tuple, Union

import numpy as np
from attr import attrs

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.numercial_methods.integrators import (
    ParamsIntegrators,
    generator_integrator,
)


@attrs(auto_attribs=True)
class ParamsIntegratorLD:
    """
    Parámetros de los métodos de integración de los descriptores lagrangianos.

    Parameters
    ----------
    diff_system: DiffSystem
        Sistema diferencial.
    t0: Union[int, float]
        Instante inicial de tiempo.
    t: Union[int, float]
        Instante de tiempo. El intevalo de tiempo de la integral
        se calculará restando y sumando tau a este instante de tiempo t.
    x: np.ndarray
        Valor de x de I(x).
    tau: Union[float, int, Tuple[float, float]]
        Valor utilizado para construir el intervalo de integración.
        Si es int o float el intervalo se calcula como [t-tau, t+tau].
        Si es una tupla de dos valores se calcula como [t-tau[0], t+tau[1]].
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Dentro de estos parámetros se incluyen el diccionario params_integrator,
        que indica los parámetros del integrador utilizado en el cálculo de
        el descriptor lagrangiano para el método ld_method = "integrate".
    args: List[Any], default None
        Argumentos de la función del sistema.

    """

    diff_system: DiffSystem
    t0: Union[int, float]
    t: Union[int, float]
    x: np.ndarray
    tau: Union[float, int, Tuple[float, float]]
    params_solver: Dict[str, any] = None
    args: List[Any] = None


def integrate_func_diff_system(ld_method: str, params_integrator: ParamsIntegratorLD) -> float:
    """
    Función que calcula el descriptor lagrangiano según el método indicado y los parámetros
    proporcionados.

    Parameters
    ----------
    ld_method: str
        Nombre del método utilizado en la integración del descriptor lagrangiano.
    params_integrator: ParamsIntegratorLD
        Parámetros del método de integración del descriptor lagrangiano.

    Returns
    -------
    mean_velocity: float
        Valor de la integral de la norma de f(t,x), dividido por la longitud del intervalo de
        integración , lo cual es similar a calcular la velocidad media
        del camino recorrido entre los límites de la integración.
    """

    dic_methods = {
        "integrate": integrate_lagrangian_descriptor,
        "differential_equations": integrate_differential_equations,
    }
    try:
        method = dic_methods[ld_method]
    except KeyError:
        raise ValueError(
            f"El método {ld_method} no está implementado, pruebe con alguno"
            f" de los métodos {list(dic_methods.keys())}"
        )
    return method(params_integrator)


def integrate_lagrangian_descriptor(ld_params: ParamsIntegratorLD) -> float:
    """
    Función que calcula la integral de norma de la función f(t,x),
    del sistema diferencia dx = f(t,x) dt, entre los instantes
    t-tau y t+tau. Es decir, calcula la integral
        I(x) = int_{t-tau[0]}^{t+tau[1]} |f(t,x)| dt.

    Parameters
    ----------
    ld_params: ParamsIntegratorLD
        Parámetros del integrador de descriptores lagrangianos.

    Returns
    -------
    mean_velocity: float
        Valor de la integral de la norma de f(t,x), dividido por la longitud del intervalo de
        integración , lo cual es similar a calcular la velocidad media
        del camino recorrido entre los límites de la integración.
    """
    x0 = ld_params.x
    if len(x0.shape) < 2:
        x0 = x0.reshape(1, x0.size)

    args = ld_params.args
    if ld_params.args is None:
        args = ()

    params_solver = {"solver_method": "solve_ivp"}
    if ld_params.params_solver is not None:
        params_solver.update(ld_params.params_solver)

    params_integrator = params_solver.pop("params_integrator", None)
    if params_integrator is None:
        params_integrator = {"method": "quad"}
    params = params_integrator.copy()
    method_integrate = params.pop("method", "quad")

    params_solver.update({"args": ld_params.args})
    limits = [ld_params.t - 1e-7, ld_params.t + 1e-7]
    if isinstance(ld_params.tau, (float, int)):
        limits = [ld_params.t - ld_params.tau, ld_params.t + ld_params.tau]
    if isinstance(ld_params.tau, tuple):
        limits = [ld_params.t - ld_params.tau[0], ld_params.t + ld_params.tau[1]]
    t_span = [ld_params.t0, limits[1]]
    delta_t = limits[1] - limits[0]
    _, _ = ld_params.diff_system.solve_function(t_span=t_span, x0=x0[0, :], **params_solver)
    function = ld_params.diff_system.get_fun_to_solve()
    x_func = ld_params.diff_system.variable.solution

    def abs_f_function(t_values) -> np.ndarray:
        if isinstance(t_values, np.ndarray):
            result = np.array(
                [np.linalg.norm(function(tv, x_func(tv), *args), ord=2) for tv in t_values]
            )
            return result
        return np.linalg.norm(function(t_values, x_func(t_values), *args), ord=2)

    params_integrator = ParamsIntegrators(abs_f_function, limits[0], limits[1], (), params)
    length_value, _ = generator_integrator(method_integrate)(params_integrator)
    mean_velocity = length_value / delta_t
    return mean_velocity


def integrate_differential_equations(ld_params: ParamsIntegratorLD) -> float:
    """
    Función que calcula la integral de norma de la función f(t,x),
    del sistema diferencia dx = f(t,x) dt, entre los instantes
    t-tau y t+tau. Es decir, calcula la integral
        I(x) = int_{t-tau[0]}^{t+tau[1]} |f(t,x)| dt.

    Parameters
    ----------
    ld_params: ParamsIntegratorLD
        Parámetros del integrador de descriptores lagrangianos.

    Returns
    -------
    mean_velocity: float
        Valor de la integral de la norma de f(t,x), dividido por la longitud del intervalo de
        integración , lo cual es similar a calcular la velocidad media
        del camino recorrido entre los límites de la integración.
    """
    x0 = ld_params.x
    if len(x0.shape) < 2:
        x0 = x0.reshape(1, x0.size)

    args = ld_params.args
    if ld_params.args is None:
        args = ()

    params_solver = {"solver_method": "solve_ivp"}
    if ld_params.params_solver is not None:
        params_solver.update(ld_params.params_solver)

    limits = [ld_params.t - 1e-7, ld_params.t + 1e-7]
    if isinstance(ld_params.tau, (float, int)):
        limits = [ld_params.t - ld_params.tau, ld_params.t + ld_params.tau]
    if isinstance(ld_params.tau, tuple):
        limits = [ld_params.t - ld_params.tau[0], ld_params.t + ld_params.tau[1]]

    t_span = [ld_params.t0, max(limits)]

    def func_system_l(v_l: DiffVariable):
        v = DiffVariable(name="v", values=v_l.values[:, :-1])
        result = ld_params.diff_system.function(v, *args)
        abs_t = np.linalg.norm(ld_params.diff_system.function(v, *args), ord=2, axis=1)
        diff_l = abs_t.reshape(1, -1)
        return np.concatenate((result, diff_l), axis=1)

    x_l = DiffVariable(name="x_l")
    diff_system_l = DiffSystem(x_l, func_system_l)
    x0_l = np.concatenate((x0[0, :], np.array([0])))
    _, _ = diff_system_l.solve_function(t_span=t_span, x0=x0_l, **params_solver)
    length_0 = diff_system_l.variable.solution(limits[0])[-1]
    length_1 = diff_system_l.variable.solution(limits[1])[-1]
    mean_velocity = (length_1 - length_0) / (limits[1] - limits[0])
    return mean_velocity
