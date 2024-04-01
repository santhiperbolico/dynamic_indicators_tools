from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    GENERALS_PARAMS,
    Param,
    ParamType,
    format_bounds_variables,
    format_n_xgrid,
    format_symbolic_number,
    format_system_projections,
)

LD_PARAMS = GENERALS_PARAMS + [
    Param("t", ParamType.INDICATOR, format_function=format_symbolic_number),
    Param("x0_min", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("x0_max", ParamType.INDICATOR, format_function=format_bounds_variables),
    Param("n_xgrid", ParamType.INDICATOR, 200, format_function=format_n_xgrid),
    Param("n_jobs", ParamType.INDICATOR, 1),
    Param("projection_config", ParamType.INDICATOR, {}, format_system_projections),
    Param("tau", ParamType.INDICATOR, 1),
    Param("method_integrate", ParamType.INDICATOR, "quad"),
    Param("log_scale_color", ParamType.SYSTEM, False),
    Param("params_solver", ParamType.INDICATOR, {}),
]
