from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .poincare_utils import get_poincare_grid_method
from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    GENERALS_PARAMS,
    Param,
    ParamType,
    format_args_system,
    format_bounds_variables,
    format_symbolic_number,
    import_string,
)


def poincare_initial_conditions_function_straight(
    x_min: Union[Sequence[float], np.ndarray],
    x_max: Union[Sequence[float], np.ndarray],
    n_points: int,
) -> List[np.ndarray]:
    """
    Función que calcula las condiciones iniciales en una recta definida por los puntos
    x_min y x_max.

    Parameters
    ----------
    x_min: Union[Sequence[float], np.ndarray]
        Límite inferior de los valores de las condiciones iniciales.
    x_max: Union[Sequence[float], np.ndarray]
        Límite superior de los valores de las condiciones iniciales.
    n_points: int
        Número de puntos a generar.

    Returns
    -------
    lis_x0: list[np.ndarray]
        Lista de condiciones iniciales aleatorias.

    """
    x_min = np.array(format_bounds_variables(x_min))
    x_max = np.array(format_bounds_variables(x_max))
    steps = np.linspace(0, 1, n_points)
    return [(x_max - x_min) * step + x_min for step in steps]


def poincare_initial_conditions_function_random(
    x_min: Union[Sequence[float], np.ndarray],
    x_max: Union[Sequence[float], np.ndarray],
    n_points: int,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Función que genera n_points de condiciones iniciales.

    Parameters
    ----------
    x_min: Union[Sequence[float], np.ndarray]
        Límite inferior de los valores de las condiciones iniciales.
    x_max: Union[Sequence[float], np.ndarray]
        Límite superior de los valores de las condiciones iniciales.
    n_points: int
        Número de puntos a generar.
    seed: Optional[int] = None
        Semilla de los números aleatorios.

    Returns
    -------
    lis_x0: list[np.ndarray]
        Lista de condiciones iniciales aleatorias.
    """
    x_min = np.array(format_bounds_variables(x_min))
    x_max = np.array(format_bounds_variables(x_max))
    if seed:
        np.random.seed(seed)
    x0_sample = np.random.uniform(low=x_min, high=x_max, size=(n_points, x_min.size))
    return [x0_sample[i, :] for i in range(n_points)]


def format_x0_grid_variables(x0_grid: List[List[Union[int, float, str]]]) -> List[np.ndarray]:
    """
    Función que formatea los límites en el espacio de fase del sistema.

    Parameters
    ----------
    x0_grid: List[List[Union[int, float, str]]]
        Lista de condiciones iniciales a formatear

    Returns
    -------
    : List[np.ndarray]
        Lista de condiciones iniciales a formateada

    """
    return [format_bounds_variables(x) for x in x0_grid]


def format_function_generator(function_generator: Optional[str] = None) -> Optional[Callable]:
    """
    Función que formatea el valor de function_generator.

    Parameters
    ----------
    function_generator: str, default None
        Si el valor es random o straight utiliza alguna de las funciones predefinidas. También
        se le puede pasar la ruta completa de una función custom que utilizar.

    Returns
    -------
    function: Optional[Callable]
        Función generadora de condiciones iniciales.
    """
    dic = {
        "random": poincare_initial_conditions_function_random,
        "straight": poincare_initial_conditions_function_straight,
    }
    if function_generator is None:
        return None
    if function_generator in dic:
        return dic[function_generator]
    return import_string(function_generator)


POINCARE_PARAMS = GENERALS_PARAMS + [
    Param("t", ParamType.SYSTEM, format_function=format_symbolic_number),
    Param(
        "poincare_method", ParamType.SYSTEM, "PoincareSectionInterpolate", get_poincare_grid_method
    ),
    Param("poincare_map", ParamType.INDICATOR, "poincare_map_function", import_string),
    Param("t_span", ParamType.INDICATOR),
    Param("x0_grid", ParamType.INDICATOR, [], format_function=format_x0_grid_variables),
    Param("n_points", ParamType.INDICATOR, 100),
    Param("args", ParamType.INDICATOR, [], format_args_system),
    Param("pm_args", ParamType.INDICATOR, [], format_args_system),
    Param("params_root", ParamType.INDICATOR, {}),
    Param("params_solver", ParamType.INDICATOR, {}),
    Param("function_generator", ParamType.INITIALS_CONDITIONS, None, format_function_generator),
    Param("params_function", ParamType.INITIALS_CONDITIONS, {}),
]
