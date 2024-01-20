from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from dynamic_indicators_tools.differentials_systems.data_transformations import (
    Projection,
    project_grid_data,
)
from dynamic_indicators_tools.differentials_systems.diff_system import (
    DoesntCoincideDimension,
    FlowMap,
)
from dynamic_indicators_tools.numercial_methods.differentiation import (
    diff_partials_grid,
    jacobian_matrix,
)


class ResultFtleMethods(NamedTuple):
    """
    Resultado que tiene que devolver un método
    que calcula los FTLE para una malla de puntos.

    Parameters
    ----------
    grid_points: Sequence[np.ndarray]
        Lista con la malla para cada dimensión. Cada elemento
        es n array de dimensión (nx_grid,..., <n_variables>, ...,nx_grid).
    ftle_grid: np.ndarray:
        Array que devuelve lso FTLE para cada punto de la malla generada en el tiempo
        t.
    """

    grid_points: Sequence[np.ndarray]
    ftle_grid: np.ndarray


def ftle_fun(
    flow: FlowMap,
    t: Union[int, float],
    x: np.ndarray,
    h_steps: Union[int, float, np.ndarray],
    params_t_close: Dict[str, Any],
) -> float:
    """
    Función que calcula el FTLE para el punto x, en el instante t para el flujo
    flow.

    Parameters
    ----------
    flow: FlowMap
        Flujo del sistema diferencial.
    t: Union[int, float]
        Instante de tiempo del FTLE.
    x: np.ndarray
        Punto condición inicial del flujo flow.
    h_steps: Union[int, float, np.ndarray]
        Pasos para las derivadas numéricas.

    Returns
    -------
    ftle: float
        FTLE del flujo flow.
    """
    x0 = x
    if len(x0.shape) < 2:
        x0 = x0.reshape(1, x0.size)

    t_close = t
    if params_t_close:
        t_close = flow.get_time_close(t, x0, **params_t_close)

    def flow_function(x_values: np.ndarray) -> np.ndarray:
        return flow(t_close, x_values)

    jacobian = jacobian_matrix(flow_function, x0, h_steps)
    delta_t = np.abs(t_close - flow.t0)
    cauchy_green = np.dot(jacobian.T, jacobian)
    return np.log(np.max(np.linalg.eig(cauchy_green)[0])) / 2 / delta_t


def ftle_element_wise(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: Union[int, np.ndarray],
    h_steps: Union[int, float, np.ndarray] = None,
    params_t_close: Dict[str, Any] = None,
    n_jobs: int = -1,
    projection_config: Dict[int, Projection] = None,
) -> ResultFtleMethods:
    """
    Función que calcula los FTLE punto por punto para una malla de puntos delimitada por
    x_min y x_max. Esta malla toma para cada dimensión n_grid_x puntos y un paso h_stpes
    para cada derivada numérica asociada a cada dimensión de la variable.

    Parameters
    ----------
    flow: FlowMap
        Flujo del cual se calcula los ftle.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    nx_grid: Union[int, np.ndarray]
        Número de puntos generados por variable, donde el número
        total de puntos será nx_grid^(n_variables.).  Si pasamos
        un entero se aplicará a todas las dimensiones. Si es una
        secuencia se aplicará según el orden de las dimensiones.
    h_steps: Union[int, float, np.ndarray], default None
        Array que indica el paso de las derivadas numéticas para cada dimensión
        de la variable. En el caso de indicar None se usará el paso de la malla.
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultFtleMethods
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
    if params_t_close is None:
        params_t_close = {}

    if h_steps is None:
        h_steps = (x0_max - x0_min) / n_xgrid
    if n_jobs < 2:
        return _ftle_map_non_paralelizable(
            flow, t, x0_min, x0_max, n_xgrid, h_steps, params_t_close, projection_config
        )
    return _ftle_map_paralelizable(
        flow, t, x0_min, x0_max, n_xgrid, h_steps, params_t_close, n_jobs, projection_config
    )


def _ftle_map_non_paralelizable(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: np.ndarray,
    h_steps: Union[int, float, np.ndarray],
    params_t_close: Dict[str, Any] = None,
    projection_config: Dict[int, Projection] = None,
) -> ResultFtleMethods:
    """
    Método no paralelizable de ftle_element_wise

    Parameters
    ----------
    flow: FlowMap
        Flujo del cual se calcula los ftle.
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
    h_steps: Union[int, float, np.ndarray], default None
        Array que indica el paso de las derivadas numéticas para cada dimensión
        de la variable. En el caso de indicar None se usará el paso de la malla.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultFtleMethods
    """
    n_var = x0_max.size
    grid_points = np.meshgrid(
        *[np.linspace(x0_min[i], x0_max[i], n_xgrid[i]) for i in range(n_var)]
    )
    grid_points = project_grid_data(grid_points, projection_config)
    ftle_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    pbar = tqdm(total=grid_points[0].size)
    while not it.finished:
        x0 = np.array([gp[it.multi_index] for gp in grid_points])
        if np.isnan(x0).any():
            ftle_grid[it.multi_index] = np.nan
        else:
            ftle_grid[it.multi_index] = ftle_fun(flow, t, x0, h_steps, params_t_close)
        _ = it.iternext()
        pbar.update(1)
    result = ResultFtleMethods(grid_points, ftle_grid)
    return result


def _ftle_map_paralelizable(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: np.ndarray,
    h_steps: Union[int, float, np.ndarray],
    params_t_close: Dict[str, Any],
    n_jobs: int,
    projection_config: Dict[int, Projection] = None,
) -> ResultFtleMethods:
    """
    Método paralelizable de ftle_element_wise

    Parameters
    ----------
    flow: FlowMap
        Flujo del cual se calcula los ftle.
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
    h_steps: Union[int, float, np.ndarray], default None
        Array que indica el paso de las derivadas numéticas para cada dimensión
        de la variable. En el caso de indicar None se usará el paso de la malla.
    n_jobs: int
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultFtleMethods
    """
    n_var = x0_max.size
    grid_points = np.meshgrid(
        *[np.linspace(x0_min[i], x0_max[i], n_xgrid[i]) for i in range(n_var)]
    )
    grid_points = project_grid_data(grid_points, projection_config)
    ftle_grid = np.zeros(grid_points[0].shape)
    it = np.nditer(grid_points[0], flags=["multi_index"])
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        multi_index_list: List[Sequence[int]] = []
        while not it.finished:
            x0 = np.array([gp[it.multi_index] for gp in grid_points])
            if np.isnan(x0).any():
                future = executor.submit(lambda x: np.nan, x0)
            else:
                future = executor.submit(ftle_fun, flow, t, x0, h_steps, params_t_close)
            futures.append(future)
            multi_index_list.append(it.multi_index)
            _ = it.iternext()
        for i, future in enumerate(tqdm(futures)):
            ftle_grid[multi_index_list[i]] = future.result()
    result = ResultFtleMethods(grid_points, ftle_grid)
    return result


def diff_flow_grid(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: Union[int, np.ndarray],
    n_jobs: int = -1,
    projection_config: Dict[int, Projection] = None,
) -> Tuple[Union[Sequence[np.ndarray], str], Sequence[np.ndarray]]:
    """
    Método que crea una malla de condiciones iniciales, donde los
    limítes están definidos por x0_min_grid y x0_max_grid y la cantidad
    de puntos son nx_grid^(numero de variables), y calcula el valor de las
    derivadas parciales del flujo para estos puntos. De vuelva la malla
    creada y una lista donde cada i-elemento es la derivada parcial
    i-ésima.

    Parameters
    ----------
    flow: FlowMap
        Flujo el cual queremos calcular las derivadas parciales.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: Union[int, Sequence[int]]
        Número de puntos generados por variable, donde el número
        total de puntos será nx_grid^(n_variables.).  Si pasamos
        un entero se aplicará a todas las dimensiones. Si es una
        secuencia se aplicará según el orden de las dimensiones.
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    grid_points: Sequence[np.ndarray]
        Lista con la malla para cada dimensión. Cada elemento
        es n array de dimensión (nx_grid,..., <n_variables>, ...,nx_grid).
    list_df: List[np.ndarray]:
        Lista de arrays de dimensión (nx_grid,..., <n_variables>, ...,nx_grid, n_variables),
        donde el elemento i de la lista representa el valor de la dereivada parcial de la
        variable x_i.
    """
    n_var = x0_max.size
    if x0_min.shape != x0_max.shape:
        raise DoesntCoincideDimension(
            "La dimensión de x0_min_grid y x0_max_grid" " deben ser iguales."
        )
    if isinstance(n_xgrid, int):
        n_xgrid = np.ones(n_var).astype(int) * n_xgrid
    nx_grid_1 = n_xgrid == 1
    h = (x0_max - x0_min) / (n_xgrid - 1)
    h[nx_grid_1] = (x0_max[nx_grid_1] - x0_min[nx_grid_1]) / (n_xgrid[nx_grid_1])
    grid_points, zz = flow.flow_grid(
        t=t,
        x0_min=x0_min - h,
        x0_max=x0_max + h,
        n_xgrid=n_xgrid + 2,
        n_jobs=n_jobs,
        projection_config=projection_config,
    )
    diff_partials = diff_partials_grid(zz, n_var, h, edge_remove=True)
    mask = (np.s_[1:-1],) * n_var
    return [gp[mask] for gp in grid_points], diff_partials


def ftle_grid(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: Union[int, np.ndarray],
    n_jobs: int = -1,
    projection_config: Dict[int, Projection] = None,
) -> ResultFtleMethods:
    """
    Función que crea una malla de condiciones iniciales, donde los
    limítes están definidos por x0_min_grid y x0_max_grid y la cantidad
    de puntos son nx_grid^(numero de variables), y calcula el valor de las
    Finite Time Lyapunov Exponent (FTLE) del flujo para cada calro de la malla en
    el tiempo t.

    Parameters
    ----------
    flow: FlowMap
        Flujo del cual se calcula los ftle.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: Union[int, Sequence[int]]
        Número de puntos generados por variable, donde el número
        total de puntos será nx_grid^(n_variables.). Si pasamos
        un entero se aplicará a todas las dimensiones. Si es una
        secuencia se aplicará según el orden de las dimensiones.
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultFtleMethods
    """
    grid_points, diff_flow = diff_flow_grid(
        flow, t, x0_min, x0_max, n_xgrid, n_jobs, projection_config
    )
    ftle_grid = np.zeros(diff_flow[0].shape[:-1])
    delta_t = abs(t - flow.t0)
    for index in np.ndindex(diff_flow[0].shape[:-1]):
        jacobian = np.concatenate([di[index].reshape(1, -1) for di in diff_flow], 0)
        cauchy_green = np.dot(jacobian.T, jacobian)
        if np.isnan(jacobian).any():
            ftle_grid[index] = np.nan
            continue
        ftle_grid[index] = np.log(np.linalg.eigvals(cauchy_green).max()) / (2 * delta_t)
    result = ResultFtleMethods(grid_points, ftle_grid)
    return result


def ftl_variational_equations(
    flow: FlowMap,
    t: Union[int, float],
    x0_min: np.ndarray,
    x0_max: np.ndarray,
    n_xgrid: Union[int, np.ndarray],
    n_jobs: int = -1,
    projection_config: Dict[int, Projection] = None,
) -> ResultFtleMethods:
    """
    Función que crea una malla de condiciones iniciales, donde los
    limítes están definidos por x0_min_grid y x0_max_grid y la cantidad
    de puntos son nx_grid^(numero de variables), y calcula el valor de las
    Finite Time Lyapunov Exponent (FTLE) del flujo para cada valor de la malla en
    el tiempo t utilizando las ecuaciones variacionales.

    Parameters
    ----------
    flow: FlowMap
        Flujo del cual se calcula los ftle.
    t: Union[int, float]
        Valor del tiempo.
    x0_min: np.ndarray
        Array de dimensión (n_variables,) que indica el valor inferior
        de la malla.
    x0_max: np.ndarray
        Array de dimensión (n_variables,) que indica el valor superior
        de la malla.
    n_xgrid: Union[int, Sequence[int]]
        Número de puntos generados por variable, donde el número
        total de puntos será nx_grid^(n_variables.). Si pasamos
        un entero se aplicará a todas las dimensiones. Si es una
        secuencia se aplicará según el orden de las dimensiones.
    n_jobs: int, default 0
        Número máximo de jobs en la paralelización.
    projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

    Returns
    -------
    result: ResultFtleMethods
    """
    n_var = x0_max.size
    if x0_min.shape != x0_max.shape:
        raise DoesntCoincideDimension(
            "La dimensión de x0_min_grid y x0_max_grid" " deben ser iguales."
        )
    if isinstance(n_xgrid, int):
        n_xgrid = np.ones(n_var).astype(int) * n_xgrid

    x_min_grid = np.concatenate((x0_min, np.eye(n_var).reshape(n_var**2)))
    x_max_grid = np.concatenate((x0_max, np.eye(n_var).reshape(n_var**2)))
    n_xgrid = np.concatenate((n_xgrid, np.ones(n_var**2))).astype(int)
    grid_points, flow_points = flow.flow_grid(
        t=t,
        x0_min=x_min_grid,
        x0_max=x_max_grid,
        n_xgrid=n_xgrid,
        n_jobs=n_jobs,
        projection_config=projection_config,
    )
    ftle_grid = np.zeros(flow_points.shape[:n_var])
    delta_t = abs(t - flow.t0)
    for index in np.ndindex(flow_points.shape[:n_var]):
        jacobian = flow_points[index].reshape(n_var + n_var**2)
        jacobian = jacobian[n_var:].reshape((n_var, n_var))
        cauchy_green = np.dot(jacobian.T, jacobian)
        if np.isnan(jacobian).any():
            ftle_grid[index] = np.nan
            continue
        if (np.abs(cauchy_green) == np.inf).any():
            ftle_grid[index] = np.inf
            continue
        ftle_grid[index] = np.log(np.abs(np.linalg.eigvals(cauchy_green)).max()) / (2 * delta_t)
    mask_max = ~(np.isnan(ftle_grid) | (ftle_grid == np.inf))
    ftle_grid[ftle_grid == np.inf] = ftle_grid[mask_max].max() * 1.10
    grid_points = [grid_points[i].reshape(ftle_grid.shape) for i in range(n_var)]
    result = ResultFtleMethods(grid_points, ftle_grid)
    return result
