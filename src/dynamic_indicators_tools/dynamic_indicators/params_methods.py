import enum
import sys
from importlib import import_module
from typing import Any, Callable, Dict, List, Sequence, Union

import attr
import numpy as np

from dynamic_indicators_tools.differentials_systems.data_transformations import Projection
from dynamic_indicators_tools.differentials_systems.diff_system import EquationFunction

SENTINEL = object()


class ParamType(enum.Enum):
    """
    Clase para definir un tipo de Param Object
    """

    SYSTEM = 1
    INDICATOR = 2
    EXECUTE = 3
    INITIALS_CONDITIONS = 4


@attr.s(frozen=True)
class Param:
    """
    Clase usada para definir un parámetro de modelos de clasificación.

    Parameters
    ----------
    name: str
        Nombre del parámetro a ser utilizado.
    param_type: ParamType
        Tipo del parámetro definido por ParamType.
    default_value: Any
        Valor por defecto para el parámetro.
    format_function: Callable[Any]
        Función de formateo asociada al parámetro.

    """

    name = attr.ib(type=str)
    param_type = attr.ib(type=ParamType)
    default_value = attr.ib(type=Any, default=SENTINEL)
    format_function = attr.ib(type=Callable, default=lambda x: x)

    def get_formatted_value(self, value: Any) -> Any:
        """
        Formatea el valor de value en el caso de que el objecto tenga definida
        una format_function. En caso contrario devuelve value tal cual se ha
        recibido.

        Parameters
        ----------
        value: Any

        Return
        ------
        Any
            El valor formateado por el método `format_function`
        """
        return self.format_function(value) if self.format_function else value


class ParamProcessor:
    """
    Clase de objetos cuya tarea es la validación de parámetros de los
    modelos de clasificación de t2o.

    Parameter
    ---------
    params_to_process: Sequence[Param]
        Lista de Param a validar.
    param: Dict[str, Any]
            Diccionario con los paraámetror
    """

    def __init__(self, params_to_process: Sequence[Param], params: Dict[str, Any]):
        self._params_to_process = params_to_process
        complete_params = self.get_default_values()
        complete_params.update(params)
        self._params = {
            p.name: p.get_formatted_value(complete_params[p.name]) for p in self._params_to_process
        }

    def get_param(self, p_name: str) -> Any:
        return self._params.get(p_name, None)

    def update_param(self, p_name: str, p_value: Any) -> Any:
        search_param = [p for p in self._params_to_process if p.name == p_name]
        if len(search_param) > 0:
            param = search_param[0]
            param_dict = {param.name: param.get_formatted_value(p_value)}
            self._params.update(param_dict)

    def get_params_by_type(self, param_type: ParamType) -> Dict[str, Any]:
        """
        Método que devuelve los parámetros según si sin válidos y por tipo.

        Parameters
        ----------
        param_type: ParamType
            Tipo de parámetro que se quiere devolver.

        Returns
        --------
        : Dict[str,Any]
            Parámetros filtrados y validados.
        """
        return {
            p.name: self._params[p.name]
            for p in self._params_to_process
            if p.param_type == param_type and p.name in self._params
        }

    def get_default_values(self) -> Dict[str, Any]:
        """
        Devuelve los parámetros que tengan valores por defecto definidos.

        Returns
        -------
        Dict[str, Any]
            Parámetros con valores por defecto definido.
        """
        return {
            param.name: param.default_value
            for param in self._params_to_process
            if not param.default_value == SENTINEL
        }


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


def format_system_function(function_path: str) -> EquationFunction:
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
    return import_string(function_path)


def format_args_system(args_system: List[str]) -> Sequence[float]:
    """
    Función que formatea lso argumentos del sistema diferencial.

    Parameters
    ----------
    args_system: List[str]
        Lista con los argumentos de la función del sistema diferencial.

    Returns
    -------
    args_system: Sequence[float]
        Secuencia con los argumentos formateados.
    """
    return tuple([format_symbolic_number(param) for param in args_system])


def format_bounds_variables(x: List[Union[int, float, str]]) -> np.ndarray:
    """
    Función que formatea los límites en el espacio de fase del sistema.

    Parameters
    ----------
    x: np.ndarray
        Límite del espacio de fase.

    Returns
    -------
    x: np.ndarray
        Límite del espacio de fase formateado.

    """
    return np.array([format_symbolic_number(v) for v in x])


def format_system_projections(projection_config: Dict[str, Any]) -> Dict[int, Projection]:
    """
    Función que dado un diccionario, donde por cada eje a proyectar se tiene
    un diccionario con la configuración de la proyección, crea un diccionario
    con cada Projection.

    Parameters
    ----------
    projection_config: Dict[str, Any]
        Diccionario donde cada clave define el eje a proyectar y sus valores
        un diccionario con la config de la proyección.

    Returns
    -------
    projection_config_new: Dict[int, Projection]
        Diccionario donde cada clave define el eje a proyectar y sus valores
        son los objetos de Projection.

    """
    projection_config_new = {}

    for key, projection in projection_config.items():
        func_cof = projection.get("function")
        function_proj_gen = import_string(func_cof.get("name"))
        args_function = func_cof.get("args", ())

        new_proj = {}
        new_proj["function"] = function_proj_gen(*args_function)
        new_proj["index_variables"] = projection.get("index_variables")
        projection_config_new[int(key)] = Projection(**new_proj)

    return projection_config_new


def format_axis_system(axis: Sequence[int] = None) -> tuple:
    """
    Función que formatea el parámetro de ejes del sistema diferencial.

    Parameters
    ----------
    axis: Sequence[int], default None
        Lista de enteros con los ejes que sequieren presentar en las imágenes.

    Returns
    -------
    axis: Tuple[int, int]


    """
    if len(axis) == 2:
        return tuple(axis)
    raise ValueError("El elemento axis debe de tener una dimensión de dos elementos,")


def format_n_xgrid(n_xgrid: Union[int, list]) -> Union[int, np.ndarray]:
    if isinstance(n_xgrid, list):
        return np.array(n_xgrid)
    return n_xgrid


GENERALS_PARAMS = [
    Param("function", ParamType.SYSTEM, format_function=format_system_function),
    Param("path", ParamType.SYSTEM, default_value="."),
    Param("system_name", ParamType.SYSTEM),
    Param("axis", ParamType.SYSTEM, [0, 1], format_axis_system),
    Param("args_system", ParamType.SYSTEM, [], format_function=format_args_system),
    Param("t0", ParamType.SYSTEM, default_value=0, format_function=format_symbolic_number),
    Param("solver_method", ParamType.SYSTEM, "solve_ivp"),
    Param("execute", ParamType.EXECUTE, default_value=False),
]
