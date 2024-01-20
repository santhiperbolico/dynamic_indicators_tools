from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from dynamic_indicators_tools.differentials_systems.data_transformations import (
    Projection,
    project_grid_data,
)
from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DoesntCoincideDimension,
)
from dynamic_indicators_tools.numercial_methods.integrators import (
    ParamsIntegrators,
    generator_integrator,
)


class ResultLDMethods(NamedTuple):
    """
    Resultado que tiene que devolver un método que calcula
    la integral, para una malla de puntos, de la norma de
    de f(t,x) del sistema diferencia:
        dx/dt = f(t,x)

    Parameters
    ----------
    grid_points: Sequence[np.ndarray]
        Lista con la malla para cada dimensión. Cada elemento
        es n array de dimensión (nx_grid,..., <n_variables>, ...,nx_grid).
    integrate_values_grid: np.ndarray:
        Array que devuelve los valores de la integral de la norma
        de f(t,x)
    """

    grid_points: Sequence[np.ndarray]
    integrate_values_grid: np.ndarray


def integrate_func_diff_system(
    diff_system: DiffSystem,
    t: Union[int, float],
    x: np.ndarray,
    tau: Union[float, int, Tuple[float, float]],
    method_integrate: str = "quad",
    params_solver: Dict[str, any] = None,
    opts_integrate: Dict[str, Any] = None,
) -> Tuple[float, float]:
    """
    Función que calcula la integral de norma de la función f(t,x),
    del sistema diferencia dx = f(t,x) dt, entre los instantes
    t-tau y t+tau. Es decir, calcula la integral
        I(x) = int_{t-tau[0]}^{t+tau[1]} |f(t,x)| dt.

    Parameters
    ----------
    diff_system: DiffSystem
        Sistema diferencial.
    t: Union[int, float]
        Instante de tiempo. El intevalo de tiempo de la integral
        se calculará restando y sumando tau a este instante de tiempo t.
    x: np.ndarray
        Valor de x de I(x).
    tau: Union[float, int, Tuple[float, float]]
        Valor utilizado para construir el intervalo de integración.
        Si es int o float el intervalo se calcula como [t-tau, t+tau].
        Si es una tupla de dos valores se calcula como [t-tau[0], t+tau[1]].
    method_integrate: str, default "quad"
        Nombre del método de integración utilizado.
        Ver get_method_integrate, por defecto se utiliza
        scipy.integrate.quad.
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Por defecto se utiliza
    opts_integrate: Dict[str, Any], default None
        Diccionario con los parámetros del método de integración.
        Por defecto no se pasa ninguno

    Returns
    -------
    mean_velocity: float
        Valor de la integral de la norma de f(t,x), dividido por la longitud del intervalo de
        integración , lo cual es similar a calcular la velocidad media
        del camino recorrido entre los límites de la integración.
    error_integrate: float
        Error de la integración.
    """
    x0 = x
    if len(x0.shape) < 2:
        x0 = x0.reshape(1, x0.size)

    if opts_integrate is None:
        opts_integrate = {}
    params = opts_integrate.copy()
    args = params.pop("args_func", ())

    if params_solver is None:
        params_solver = {"solver_method": "solve_ivp", "t0": 0, "args": args}

    limits = [t - 1e-7, t + 1e-7]
    if isinstance(tau, (float, int)):
        limits = [t - tau, t + tau]
    if isinstance(tau, tuple):
        limits = [t - tau[0], t + tau[1]]
    t0 = params_solver.pop("t0", 0)
    t_span = [t0, limits[1]]
    delta_t = limits[1] - limits[0]
    _, _ = diff_system.solve_function(t_span=t_span, x0=x0[0, :], **params_solver)
    function = diff_system.get_fun_to_solve()
    x_func = diff_system.variable.solution

    def abs_f_function(t_values) -> np.ndarray:
        if isinstance(t_values, np.ndarray):
            result = np.array(
                [np.linalg.norm(function(tv, x_func(tv), *args), ord=2) for tv in t_values]
            )
            return result
        return np.linalg.norm(function(t_values, x_func(t_values), *args), ord=2)

    params_integrator = ParamsIntegrators(abs_f_function, limits[0], limits[1], (), params)
    length_value, error_length = generator_integrator(method_integrate)(params_integrator)
    mean_velocity = length_value / delta_t
    return mean_velocity, error_length


def lagrangian_descriptors(
    diff_system: DiffSystem,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: Union[int, np.ndarray],
    tau: Union[float, int, Tuple[float, float]],
    method_integrate: str,
    params_solver: Dict[str, Any] = None,
    opts_integrate: Dict[str, Any] = None,
    n_jobs: int = -1,
    projection_config: Dict[int, Projection] = None,
) -> ResultLDMethods:
    """
    Función que calcula la malla de puntos de la integral de norma de la función f(t,x),
    del sistema diferencia dx = f(t,x) dt, entre los instantes t-tau y t+tau.
    Es decir, calcula la integral:
        I(x) = int_{t-tau[0]}^{t+tau[1]} |f(t,x)| dt.

    Parameters
    ----------
    diff_system: DiffSystem
        Sistema diferencial para integrar.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: Union[int, np.ndarray]
        Número de puntos generados por variable, donde el número
        total de puntos será nx_grid^(n_variables.).  Si pasamos
        un entero se aplicará a todas las dimensiones. Si es una
        secuencia se aplicará según el orden de las dimensiones.
    tau: Union[float, int, Tuple[float, float]]
        Valor utilizado para construir el intervalo de integración.
        Si es int o float el intervalo se calcula como [t-tau, t+tau].
        Si es una tupla de dos valores se calcula como [t-tau[0], t+tau[1]].
    method_integrate: str, default "quad"
        Nombre del método de integración utilizado.
        Ver get_method_integrate, por defecto se utiliza
        scipy.integrate.quad.
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Por defecto se utiliza
    opts_integrate: Dict[str, Any], default None
        Diccionario con los parámetros del método de integración.
        Por defecto no se pasa ninguno
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
        Diccionario que recoge las proyecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = x0_max.size
    if isinstance(n_xgrid, int):
        n_xgrid = np.ones(n_var).astype(int) * n_xgrid
    if x0_min.size != n_var:
        raise DoesntCoincideDimension(
            "La dimensión de x0_min_grid y x0_max_grid" " deben ser iguales."
        )
    if n_xgrid.size != n_var:
        raise DoesntCoincideDimension(
            "La dimensión de nx_grid no coincide con el número de variables."
        )

    if n_jobs < 2:
        return _lagrangian_descriptors_non_paralelizable(
            diff_system=diff_system,
            t=t,
            x0_min=x0_min,
            x0_max=x0_max,
            n_xgrid=n_xgrid,
            tau=tau,
            method_integrate=method_integrate,
            params_solver=params_solver,
            opts_integrate=opts_integrate,
            projection_config=projection_config,
        )
    return _lagrangian_descriptors_paralelizable(
        diff_system=diff_system,
        t=t,
        x0_min=x0_min,
        x0_max=x0_max,
        n_xgrid=n_xgrid,
        tau=tau,
        method_integrate=method_integrate,
        params_solver=params_solver,
        opts_integrate=opts_integrate,
        n_jobs=n_jobs,
        projection_config=projection_config,
    )


def _lagrangian_descriptors_non_paralelizable(
    diff_system: DiffSystem,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: np.ndarray,
    tau: Union[float, int, Tuple[float, float]],
    method_integrate: str,
    params_solver: Dict[str, Any] = None,
    opts_integrate: Dict[str, Any] = None,
    projection_config: Dict[int, Projection] = None,
) -> ResultLDMethods:
    """
    Método no paralelizable de lagrangian_descriptors

    Parameters
    ----------
    diff_system: DiffSystem
        Sistema diferencial para integrar.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: np.ndarray
        Array con el número de puntos generados por variable.
    method_integrate: str, default "quad"
        Nombre del método de integración utilizado.
        Ver get_method_integrate, por defecto se utiliza
        scipy.integrate.quad.
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Por defecto se utiliza
    opts_integrate: Dict[str, Any], default None
        Diccionario con los parámetros del método de integración.
        Por defecto no se pasa ninguno
    projection_config: Dict[int, Projection], default None
        Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = x0_max.size
    grid_points = np.meshgrid(
        *[np.linspace(x0_min[i], x0_max[i], n_xgrid[i]) for i in range(n_var)]
    )
    grid_points = project_grid_data(grid_points, projection_config)
    lag_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    pbar = tqdm(total=grid_points[0].size)

    while not it.finished:
        x0 = np.array([gp[it.multi_index] for gp in grid_points])
        if np.isnan(x0).any():
            lag_grid[it.multi_index] = np.nan
        else:
            lag_grid[it.multi_index], _ = integrate_func_diff_system(
                diff_system, t, x0, tau, method_integrate, params_solver, opts_integrate
            )
        _ = it.iternext()
        pbar.update(1)
    result = ResultLDMethods(grid_points, lag_grid)
    return result


def _lagrangian_descriptors_paralelizable(
    diff_system: DiffSystem,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: np.ndarray,
    tau: Union[float, int, Tuple[float, float]],
    method_integrate: str,
    params_solver: Dict[str, Any] = None,
    opts_integrate: Dict[str, Any] = None,
    n_jobs: int = 2,
    projection_config: Dict[int, Projection] = None,
) -> ResultLDMethods:
    """
    Método paralelizable de ftle_element_wise

    Parameters
    ----------
    diff_system: DiffSystem
        Sistema diferencial para integrar.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: np.ndarray
         Array con el número de puntos generados por variable.
    method_integrate: str, default "quad"
        Nombre del método de integración utilizado.
        Ver get_method_integrate, por defecto se utiliza
        scipy.integrate.quad.
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Por defecto se utiliza
    opts_integrate: Dict[str, Any], default None
        Diccionario con los parámetros del método de integración.
        Por defecto no se pasa ninguno
    n_jobs: int
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
        Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = x0_max.size
    grid_points = np.meshgrid(
        *[np.linspace(x0_min[i], x0_max[i], n_xgrid[i]) for i in range(n_var)]
    )
    grid_points = project_grid_data(grid_points, projection_config)
    lag_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        multi_index_list: List[Sequence[int]] = []
        while not it.finished:
            x0 = np.array([gp[it.multi_index] for gp in grid_points])
            if np.isnan(x0).any():
                future = executor.submit(lambda x: np.nan, x0)
            else:
                future = executor.submit(
                    integrate_func_diff_system,
                    diff_system,
                    t,
                    x0,
                    tau,
                    method_integrate,
                    params_solver,
                    opts_integrate,
                )
            futures.append(future)
            multi_index_list.append(it.multi_index)
            _ = it.iternext()
        for i, future in enumerate(tqdm(futures)):
            lag_grid[multi_index_list[i]] = future.result()[0]
    result = ResultLDMethods(grid_points, lag_grid)
    return result
