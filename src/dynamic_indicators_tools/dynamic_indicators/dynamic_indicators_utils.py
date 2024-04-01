import logging
import os.path
from typing import Any, Dict
from warnings import warn

from .dynamic_indicators_process import DynamicIndicator
from .finite_time_lyapunov_exponents.ftle_indicator import (
    FtleElementWise,
    FtleGrid,
    FtleVariationalEquations,
)
from .lagrangian_descriptors.ld_indicator import LagrangianDescriptor
from .poincare_maps.poincare_indicator import PoincareSections

DynamicIndicatorDict = {
    FtleElementWise.name_dynamic_indicator: FtleElementWise,
    FtleGrid.name_dynamic_indicator: FtleGrid,
    LagrangianDescriptor.name_dynamic_indicator: LagrangianDescriptor,
    FtleVariationalEquations.name_dynamic_indicator: FtleVariationalEquations,
    PoincareSections.name_dynamic_indicator: PoincareSections,
}


class DynamicIndicatorNotExist(Exception):
    pass


def get_dynamic_indicator(dynamic_indicator_name: str) -> DynamicIndicator:
    """
    Función que dado un nombre del indicador devuelve el objeto DynamicIndicator
    correspondiente si existe. En caso de que no devuelve un error DynamicIndicatorNotExist

    Parameters
    ----------
    dynamic_indicator_name: str
        Nombre del indicador

    Returns
    -------
    dynamic_indicators_object: DynamicIndicator
        Objeto instanciado del indicador.
    """
    try:
        return DynamicIndicatorDict[dynamic_indicator_name]()
    except KeyError:
        raise DynamicIndicatorNotExist(
            f"El indicador {dynamic_indicator_name} no existe."
            f" Pruebe con {list(DynamicIndicatorDict.keys())}"
        )


def main_process_di(params: Dict[str, Any]) -> None:
    """
    Función que crea las gráficas de indicadores dinámicos para el problema
    configurado a través de los parámetros system y system_params. Cada indicador dinámico
    está especificado en dynamic_indicators donde la key es el nombre y los valores
    un diccionario con los parámetros del indicador. En ese diccionario debe aparecer
    un parámetro "execute" que indica si se quiere procesar dicho indicador, por defecto
    lo toma como False.

    Parameters
    ----------
    params: Dict[str, Any]
        Diccionario con las funciones del sistema diferencial
    system_params: Dict[str, Any]
        Diccionario con los parámetros generales de los indicadores y del sistema.
    dynamic_indicators: Dict[str, Any]
        Diccionario donde Cada indicador dinámico está especificado con la key como el nombre
        y los valores como un diccionario con los parámetros del indicador. En ese diccionario
        debe aparecer un parámetro "execute" que indica si se quiere procesar dicho indicador,
        por defecto lo toma como False.
    """
    dynamic_indicators = params.copy()
    system_params = dynamic_indicators.pop("system_params")
    path_system = system_params.get("path", ".")
    if not os.path.exists(path_system):
        os.mkdir(path_system)
    for dynamic_indicator_name, dynamic_indicator_params in dynamic_indicators.items():
        try:
            dynamic_indicator_object = get_dynamic_indicator(dynamic_indicator_name)
        except DynamicIndicatorNotExist:
            warn(
                f"El indicador {dynamic_indicator_name} no existe. "
                "Se continua con el siguiente indicador."
            )
            continue
        dynamic_indicator_object.process(params)
        logging.info("----------------------------------------------")
