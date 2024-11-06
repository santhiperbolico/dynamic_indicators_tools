from typing import Optional

from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    GENERALS_PARAMS,
    Param,
    ParamType,
    format_args_system,
    format_bounds_variables,
    format_n_xgrid,
    format_symbolic_number,
    format_system_projections,
)


def format_ld_method(value: Optional[str]) -> Optional[str]:
    """
    Función que controla los valores que puede tomar ld_method. Los posibles valores
    que puede tomar son "integrate", "differential_equations".

    Parameters
    ----------
    value: Optional[str]
        Nombre del método de cálculo value

    Returns
    -------
    value: Optional[str]
        Nombre del método si es válido.

    Raises
    ------
    ValueError: Si el método indicado en value no está implementado.
    """

    ld_methods = ["integrate", "differential_equations"]

    if value is None:
        return "integrate"
    if value in ld_methods:
        return value
    raise ValueError(
        f"El método de cálculo de LD {value} no está implementado."
        f"Pruebe con alguno de {ld_methods}"
    )


LD_PARAMS = GENERALS_PARAMS + [
    Param("t", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("t0", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("args", ParamType.INDICATOR, (), format_function=format_args_system),
    Param("x0_min", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("x0_max", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("n_xgrid", ParamType.INDICATOR, 200, format_function=format_n_xgrid),
    Param("n_jobs", ParamType.INDICATOR, 1),
    Param("projection_config", ParamType.INDICATOR, {}, format_system_projections),
    Param("tau", ParamType.INDICATOR, 1),
    Param("log_scale_color", ParamType.SYSTEM, False),
    Param("ld_method", ParamType.INDICATOR, None, format_ld_method),
    Param("params_solver", ParamType.INDICATOR, {}),
    Param("params_integrator", ParamType.INDICATOR, {"method": "quad"}),
]
