from typing import Any, Dict

import numpy as np

from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    GENERALS_PARAMS,
    Param,
    ParamType,
    format_bounds_variables,
    format_n_xgrid,
    format_symbolic_number,
    format_system_projections,
    import_string,
)


def format_params_t_close(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función que formatea los parámetros para el tiempo cercano en los procesoso de
    FTLE.

    Parameters
    ----------
    params: Dict[str, Any]
        Parámetros asociados a t_close.

    Returns
    -------
    params_time_close: Dict[str, Any]
        Parámetros asociados a t_close formateados.

    """
    params_time_close = None
    t_close = params.pop("t_close", False)
    if t_close:
        params_time_close = params
        if isinstance(params_time_close["mod_solution"], list):
            params_time_close["mod_solution"] = np.array(
                format_symbolic_number(x) for x in params_time_close["mod_solution"]
            )
        else:
            params_time_close["mod_solution"] = np.array(
                format_symbolic_number(params_time_close["mod_solution"])
            )
    return params_time_close


FTLE_ELEMENT_WISE_PARAMS = GENERALS_PARAMS + [
    Param("t", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("x0_min", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("x0_max", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("n_xgrid", ParamType.INDICATOR, 200, format_function=format_n_xgrid),
    Param("n_jobs", ParamType.INDICATOR, 1),
    Param("h_steps", ParamType.INDICATOR, None),
    Param("params_t_close", ParamType.INDICATOR, {"t_close": False}, format_params_t_close),
    Param("projection_config", ParamType.INDICATOR, {}, format_system_projections),
    Param("params_solver", ParamType.SYSTEM, {}),
]

FTLE_GRID_PARAMS = GENERALS_PARAMS + [
    Param("t", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("x0_min", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("x0_max", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("n_xgrid", ParamType.INDICATOR, 200, format_function=format_n_xgrid),
    Param("n_jobs", ParamType.INDICATOR, 1),
    Param("projection_config", ParamType.INDICATOR, {}, format_system_projections),
    Param("params_solver", ParamType.SYSTEM, {}),
]

FTLE_VARIATIONAL_EQUATIONS = GENERALS_PARAMS + [
    Param("t", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("x0_min", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("x0_max", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("n_xgrid", ParamType.INDICATOR, 200, format_function=format_n_xgrid),
    Param("n_jobs", ParamType.INDICATOR, 1),
    Param("var_system", ParamType.SYSTEM, format_function=import_string),
    Param("projection_config", ParamType.INDICATOR, {}, format_system_projections),
    Param("params_solver", ParamType.SYSTEM, {}),
]
