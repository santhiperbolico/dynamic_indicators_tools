from typing import List, Union

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


def format_function_generator(function_generator: str = None):
    if function_generator:
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
