from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Union

import numpy as np
from attr import attrs
from tqdm import tqdm

from dynamic_indicators_tools.differentials_systems.data_transformations import (
    Projection,
    project_grid_data,
)
from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, check_array_size
from dynamic_indicators_tools.dynamic_indicators.lagrangian_descriptors.ld_integrators import (
    ParamsIntegratorLD,
    integrate_func_diff_system,
)


@attrs(auto_attribs=True)
class ParamsLD:
    """
    Parámetros LD

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
    params_solver: Dict[str, Any], default None
        Diccionario con los parámetros del solver del sistema diferencial.
        Por defecto se utiliza
    params_integrator: Dict[str, Any] = None
        Parámetros del integrador utilizado en el cálculo del descriptor lagrangiano.
    args: List[Any], default None
        Argumentos de la función del sistema.
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
        Diccionario que recoge las proyecciones que se hagan en cada dimensión.
    """

    diff_system: DiffSystem
    t0: Union[int, float]
    t: Union[int, float]
    ld_method: str
    x0_min: np.ndarray
    x0_max: np.ndarray
    n_xgrid: Union[int, np.ndarray]
    tau: Union[float, int, Tuple[float, float]]
    params_solver: Dict[str, Any] = None
    params_integrator: Dict[str, Any] = None
    args: List[Any] = None
    n_jobs: int = -1
    projection_config: Dict[int, Projection] = None

    def get_params_integrators(self, x) -> ParamsIntegratorLD:
        return ParamsIntegratorLD(
            x=x,
            diff_system=self.diff_system,
            t0=self.t0,
            t=self.t,
            tau=self.tau,
            params_solver=self.params_solver,
            args=self.args,
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


def lagrangian_descriptors(params_ld: ParamsLD) -> ResultLDMethods:
    """
    Función que calcula la malla de puntos de la integral de norma de la función f(t,x),
    del sistema diferencia dx = f(t,x) dt, entre los instantes t-tau y t+tau.
    Es decir, calcula la integral:
        I(x) = int_{t-tau[0]}^{t+tau[1]} |f(t,x)| dt.

    Parameters
    ----------
    params_ld: ParamsLD
        Parámetros del indicador dinámico.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = params_ld.x0_max.size
    if isinstance(params_ld.n_xgrid, int):
        params_ld.n_xgrid = np.ones(n_var).astype(int) * params_ld.n_xgrid
    check_array_size({"x0_max_grid": params_ld.x0_min, "n_xgrid": params_ld.n_xgrid}, n_var)

    if params_ld.n_jobs < 2:
        return _lagrangian_descriptors_non_paralelizable(params_ld=params_ld)
    return _lagrangian_descriptors_paralelizable(params_ld=params_ld)


def _lagrangian_descriptors_non_paralelizable(params_ld: ParamsLD) -> ResultLDMethods:
    """
    Método no paralelizable de lagrangian_descriptors

    Parameters
    ----------
    params_ld: ParamsLD
        Parámetros del indicador dinámico.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = params_ld.x0_max.size
    grid_points = np.meshgrid(
        *[
            np.linspace(params_ld.x0_min[i], params_ld.x0_max[i], params_ld.n_xgrid[i])
            for i in range(n_var)
        ]
    )
    grid_points = project_grid_data(grid_points, params_ld.projection_config)
    lag_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    pbar = tqdm(total=grid_points[0].size)

    while not it.finished:
        x0 = np.array([gp[it.multi_index] for gp in grid_points])
        if np.isnan(x0).any():
            lag_grid[it.multi_index] = np.nan
        else:
            lag_grid[it.multi_index] = integrate_func_diff_system(
                params_ld.ld_method, params_ld.get_params_integrators(x0)
            )
        _ = it.iternext()
        pbar.update(1)
    result = ResultLDMethods(grid_points, lag_grid)
    return result


def _lagrangian_descriptors_paralelizable(params_ld: ParamsLD) -> ResultLDMethods:
    """
    Método paralelizable de agrangian_descriptors

    Parameters
    ----------
    params_ld: ParamsLD
        Parámetros del indicador dinámico.

    Returns
    -------
    result: ResultLDMethods
    """
    n_var = params_ld.x0_max.size
    grid_points = np.meshgrid(
        *[
            np.linspace(params_ld.x0_min[i], params_ld.x0_max[i], params_ld.n_xgrid[i])
            for i in range(n_var)
        ]
    )
    grid_points = project_grid_data(grid_points, params_ld.projection_config)
    lag_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    with ThreadPoolExecutor(max_workers=params_ld.n_jobs) as executor:
        futures = []
        multi_index_list: List[Sequence[int]] = []
        while not it.finished:
            x0 = np.array([gp[it.multi_index] for gp in grid_points])
            if np.isnan(x0).any():
                future = executor.submit(lambda x: np.nan, x0)
            else:
                future = executor.submit(
                    integrate_func_diff_system,
                    params_ld.ld_method,
                    params_ld.get_params_integrators(x0),
                )
            futures.append(future)
            multi_index_list.append(it.multi_index)
            _ = it.iternext()
        for i, future in enumerate(tqdm(futures)):
            lag_grid[multi_index_list[i]] = future.result()[0]
    result = ResultLDMethods(grid_points, lag_grid)
    return result
