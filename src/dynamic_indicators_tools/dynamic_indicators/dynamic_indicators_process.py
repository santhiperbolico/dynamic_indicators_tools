import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from attr import attrs

from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DiffVariable,
    FlowMap,
)


def create_system(system_params: Dict[str, Any]) -> FlowMap:
    """
    Función que devueve el FlowMap asociado a la función de systems.

    Parameters
    ----------
    system_params: Dict[str, Any]
        Diccionario con los parámetros generales de los indicadores y del sistema.

    Returns
    -------
    flow_map: FlowMap
        FlowMap asociado al sistema diferencial que define systems.
    """
    function_system = system_params.get("function")
    system_name = system_params.get("system_name")
    args_system = system_params.get("args_system")
    t0 = system_params.get("t0")
    solver_method = system_params.get("solver_method")

    logging.info("\t - Creando el sistema %s" % system_name)
    x = DiffVariable("x")
    flow_map = FlowMap(diff_system=DiffSystem(x, function_system), t0=t0)
    flow_map.set_params_fun_solver(solver_method=solver_method, args_func=args_system)
    return flow_map


@attrs
class DynamicIndicator(ABC):
    @abstractmethod
    def process(self, params: Dict[str, Any]) -> None:
        """
        Proceso que ejecuta el indicador dinámico
        """
        pass
