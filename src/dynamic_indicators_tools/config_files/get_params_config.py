import json
import os.path
import sys
from importlib import import_module
from typing import Any, Dict, Tuple, Union

import numpy as np

from dynamic_indicators_tools.differentials_systems.data_transformations import Projection


def format_symbolic_number(x: Union[str, float, int]) -> float:
    """
    Función que fomratea simbolos númericos en números reales,
    permitiendo utilizar explresiones como 2_*_pi para obtener 2*pi.
    Cada opteración y número se separa mediante _ y hay que tener en cuenta
    que este formateo no respeta las prioridades de las operaciones.

    Parameters
    ----------
    x: Union[str, float, int]
        Símbolo a formatear. Si es un entero o int devuelve el valor tal cual.

    Returns
    -------
    format_value: float
        Valor formateado.
    """
    symbolic_numbers = {"e": np.exp(1), "pi": np.pi, "tau": 2 * np.pi}
    symbolic_operations = {
        "*": lambda z, y: z * y,
        "-": lambda z, y: z - y,
        "+": lambda z, y: z + y,
        "/": lambda z, y: z / y,
        "^": lambda z, y: z**y,
    }
    if isinstance(x, str):
        components_x = x.split("_")
        format_value = 0.0
        op_value = symbolic_operations["+"]
        for comp_x in components_x:
            try:
                format_value = op_value(format_value, symbolic_numbers[comp_x])
                op_value = symbolic_operations["+"]
                continue
            except KeyError:
                pass
            try:
                op_value = symbolic_operations[comp_x]
                continue
            except KeyError:
                pass
            try:
                format_value = op_value(format_value, float(comp_x))
                op_value = symbolic_operations["+"]
            except ValueError:
                raise RuntimeError(f"{comp_x} no pertenece a ningún número ni operación simbólica")
        return format_value
    return x


def cached_import(module_path, class_name):
    """
    Método que comprueba si un modulo está cargado e inicializado
    """
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_string(dotted_path):
    """
    Función que dada una ruta con puntos de un módulo específico lo carga.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        ) from err


def get_system_function(system_path: str) -> Dict[str, Any]:
    """
    Método que genera el diccionario con las funcion del sistema
    y la función que genera los extremales.

    Parameters
    ----------
    system_path: str
        Ruta con puntos del archivo que contiene las funciones.

    Returns
    -------
    system: Dict[str, Any]
        Diccionario con la función del sistema y los extremales del funcional.

    """
    response = {
        "function": import_string(system_path + ".function_system"),
        "extremals": import_string(system_path + ".extremals_functionals"),
    }
    return response


def get_system_projections(
    system_path: str, projection_config: Dict[str, Any]
) -> Dict[int, Projection]:
    projection_config_new = {}

    for key, projection in projection_config.items():
        func_cof = projection.pop("function")
        function_proj_gen = import_string(system_path + f".{func_cof.get('name')}")
        args_function = func_cof.get("args", ())
        projection["function"] = function_proj_gen(*args_function)
        projection_config_new[int(key)] = Projection(**projection)

    return projection_config_new


def get_formated_params(
    system_params: Dict[str, Any], dynamic_indicators: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Función que formatea los valores de system_params y dynamic_indicators antes de
    devolverlos a la función principal.

    Parameters
    ----------
    system_params: Dict[str, Any]
        Diccionario con los parámetros de configuración del sistema dinámico y la configuraicón
        general de lso indicadores dinámicos.
    dynamic_indicators: Dict[str, Any]
        Diccionario con los parámetros de configuración de cada uno de los sistemas dinámicos.

    Returns
    -------
    system: Dict[str, Any]
        Diccionario con la función del sistema y los extremales del funcional, pero con
        los valores formateados.
    system_params: Dict[str, Any]
        Diccionario con los parámetros de configuración del sistema dinámico y la configuraicón
        general de lso indicadores dinámicos, pero con los valores formateados.

    """
    args_system = tuple(
        [format_symbolic_number(param) for param in system_params.get("args_system")]
    )
    system_name = system_params.get("system_name")
    update_system_params = {
        "system_name": f"{system_name}_{'_'.join([str(a) for a in args_system])}",
        "t0": system_params.get("t0", 0),
        "t1": format_symbolic_number(system_params.get("t1")),
        "x_min": np.array([format_symbolic_number(v) for v in system_params["x_min"]]),
        "x_max": np.array([format_symbolic_number(v) for v in system_params["x_max"]]),
        "n_xgrid": system_params.get("n_xgrid", 200),
        "n_jobs": system_params.get("n_jobs", 1),
        "solver_method": system_params.get("solver_method", "solve_ivp"),
    }
    if isinstance(update_system_params["n_xgrid"], list):
        update_system_params["n_xgrid"] = np.array(update_system_params["n_xgrid"])

    system_params.update(update_system_params)

    params_time_close = None
    if dynamic_indicators["ftle_element_wise"].get("t_close", False):
        params_time_close = dynamic_indicators["ftle_element_wise"].get("params_t_close")
        if isinstance(params_time_close["mod_solution"], list):
            params_time_close["mod_solution"] = np.array(
                format_symbolic_number(x) for x in params_time_close["mod_solution"]
            )
        else:
            params_time_close["mod_solution"] = np.array(
                format_symbolic_number(params_time_close["mod_solution"])
            )
    update_dynamic_indicators = {
        "ftle_element_wise": {
            "execute": dynamic_indicators["ftle_element_wise"].get("execute", False),
            "h_steps": dynamic_indicators["ftle_element_wise"].get("h_steps", None),
            "t_close": dynamic_indicators["ftle_element_wise"].get("t_close", False),
            "params_t_close": params_time_close,
        },
        "ftle_grid": {
            "execute": dynamic_indicators["ftle_grid"].get("execute", False),
        },
        "ftle_variational_equations": {
            "execute": dynamic_indicators["ftle_variational_equations"].get("execute", False),
            "system": dynamic_indicators["ftle_variational_equations"].get("system", None),
        },
        "lagrangian_descriptors": {
            "execute": dynamic_indicators["lagrangian_descriptors"].get("execute", False),
            "tau": dynamic_indicators["lagrangian_descriptors"].get("tau", False),
            "method_integrate": dynamic_indicators["lagrangian_descriptors"].get(
                "method_integrate", "quad"
            ),
            "log_scale_color": dynamic_indicators["lagrangian_descriptors"].get(
                "log_scale_color", False
            ),
            "plot_orbits": dynamic_indicators["lagrangian_descriptors"].get("plot_orbits", False),
        },
    }
    dynamic_indicators.update(update_dynamic_indicators)
    return system_params, dynamic_indicators


def _get_main_params(
    main_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Funcion que devuelve los parámetros del sistema a analizar desde la ruta del
    archivo de condguración parseada.

    Parameters
    -----------
    main_config: Dict[str, Any]
        Diccionario con la configuración del proyecto.

    Returns
    -------
    system: Dict[str, Any]
        Diccionario con la función del sistema y los extremales del funcional.
    system_params: Dict[str, Any]
        Diccionario con los parámetros de configuración del sistema dinámico y la configuraicón
        general de lso indicadores dinámicos.
    dynamic_indicators: Dict[str, Any]
        Configuraciones específicas de lso indicadores dinámicos.
    """
    system_params = main_config["system_params"]
    if not os.path.exists(system_params.get("path", ".")):
        os.mkdir(system_params.get("path", "."))
    system_params["axis"] = tuple(system_params.get("axis", [0, 1]))
    system_params["t1"] = format_symbolic_number(system_params.get("t1"))
    dynamic_indicators = {
        "ftle_element_wise": main_config.get("ftle_element_wise", {"execute": False}),
        "ftle_grid": main_config.get("ftle_grid", {"execute": False}),
        "lagrangian_descriptors": main_config.get("lagrangian_descriptors", {"execute": False}),
        "ftle_variational_equations": main_config.get(
            "ftle_variational_equations", {"execute": False}
        ),
    }
    system_path = system_params.pop("system_path")
    system = get_system_function(system_path)
    projection_config = system_params.get("projection_config", {})
    system_params["projection_config"] = get_system_projections(system_path, projection_config)
    system_params, dynamic_indicators = get_formated_params(system_params, dynamic_indicators)

    if dynamic_indicators["ftle_variational_equations"].get("system", None):
        system_ve_name = dynamic_indicators["ftle_variational_equations"]["system"]
        system_ve = import_string(f"{system_path}.{system_ve_name}")
        dynamic_indicators["ftle_variational_equations"]["system"] = system_ve

    return system, system_params, dynamic_indicators


def get_main_params(
    config_json_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Funcion que devuelve los parámetros del sistema a analizar desde la ruta del
    archivo de condguración parseada.

    Returns
    -------
    system: Dict[str, Any]
        Diccionario con la función del sistema y los extremales del funcional.
    system_params: Dict[str, Any]
        Diccionario con los parámetros de configuración del sistema dinámico y la configuraicón
        general de lso indicadores dinámicos.
    dynamic_indicators: Dict[str, Any]
        Configuraciones específicas de lso indicadores dinámicos.
    """
    file_c = open(config_json_path, "r")
    main_config = json.load(file_c)
    file_c.close()
    return _get_main_params(main_config)
