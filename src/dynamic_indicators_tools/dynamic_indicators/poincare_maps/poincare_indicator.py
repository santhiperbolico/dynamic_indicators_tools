import logging
import os
from typing import Any, Dict, List

from .poincare_params import POINCARE_PARAMS
from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_process import (
    DynamicIndicator,
    create_system,
)
from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    Param,
    ParamProcessor,
    ParamType,
)
from dynamic_indicators_tools.dynamic_indicators.plot_descriptors import plot_poincare_sections


def update_x0_grid(params_processor: ParamProcessor) -> ParamProcessor:
    initial_conditions = params_processor.get_params_by_type(ParamType.INITIALS_CONDITIONS)
    if initial_conditions.get("function_generator"):
        function_generator = initial_conditions.get("function_generator")
        new_x0_grid = function_generator(**initial_conditions.get("params_function"))
        params_processor.update_param("x0_grid", new_x0_grid)

    return params_processor


class PoincareSections(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de las secciones de Poincaré
    para una malla de puntos de condiciones iniciales.
    """

    name_dynamic_indicator: str = "poincare_section"
    default_params: List[Param] = POINCARE_PARAMS

    def create_params_processor(self, params: Dict[str, Any] = None) -> ParamProcessor:
        """
        Método que crea el params processor para el método de las secciones de Poincaré.

        Parameters
        ----------
        params: Dict[str, Any], default None
            Diccionario con los parámetros del indicador dinámico.

        Returns
        -------
        param_processor: ParamProcessor
            ParamProcessor del indicador dinámico.

        """
        params_indicator = {}
        if params is None:
            params = {}
        params_indicator.update(params.get("system_params"))
        params_poincare = params.get(self.name_dynamic_indicator, {}).copy()
        initial_conditions = params_poincare.pop("initial_conditions", {})
        params_indicator.update(params_poincare)
        params_indicator.update(initial_conditions)
        params_indicator.update(
            {
                "t_span": [params_indicator.get("t0"), params_indicator.get("t")],
                "args": params_indicator.get("args_system"),
            }
        )
        params_processor = ParamProcessor(self.default_params, params_indicator)
        params_processor = update_x0_grid(params_processor)
        return params_processor

    def load_params(self, params: Dict[str, Any]) -> ParamProcessor:
        """
        Método que carga los parámetros a un objeto param processor.

        Parameters
        ----------
        params: Dict[str, Any]
            Diccionario con los parámetros del proceso.

        Returns
        -------
        params_processor: ParamProcessor
            Objeto paramprocessor con los parámetros del método del indicador dinámico.

        """
        logging.info("-- Cargando parámetros.")
        params_processor = self.create_params_processor(params)
        n_points = params_processor.get_param("n_points")
        system_params = params_processor.get_params_by_type(ParamType.SYSTEM)
        params_indicator = params_processor.get_params_by_type(ParamType.INDICATOR)
        logging.info("-- Parámetros del proceso:")
        logging.info("\t T=%i" % system_params.get("t"))
        logging.info("\t Número de puntos =%i" % n_points)
        logging.info("\t Solver usado = %s " % system_params.get("solver_method"))
        logging.info("\t Las condiciones iniciales usadas son:")
        for x0 in params_indicator.get("x0_grid"):
            logging.info("\t %s " % (str(list(x0))))
        return params_processor

    def create_file_name_process(self, params_processor: ParamProcessor) -> str:
        """
        Método que crea el título del indicador dinámico con los procesos.

        Parameters
        ----------
        params_processor: ParamProcessor
            Objeto paramprocessor con los parámetros del proceso.

        Returns
        -------
        fname: str
            Nombre asociado al indicador con los prámetros y problema utilizado.

        """

        n_points = params_processor.get_param("n_points")
        system_name = params_processor.get_param("system_name")
        t = params_processor.get_param("t")
        fname = f"{system_name}_{self.name_dynamic_indicator}_t_{t:.0f}_nx_grid_{n_points:.0f}"
        return fname

    def process(self, params: Dict[str, Any]) -> None:
        """
        Proceso que ejecuta el indicador dinámico para una malla de puntos sacando
        una gráfica en el plano indicado dentro de system_params con los valores del
        indicador en colores. Guarda las gráficas en la ruta especificada.

        Parameters
        ----------
        params: Dict[str, Any]
            Parámetros del indicador dinámico.
        """
        execute = params.get(self.name_dynamic_indicator, {}).get("execute", False)
        if execute:
            logging.info(f"Ejecutando {self.name_dynamic_indicator}")
            params_processor = self.load_params(params)
            fname = self.create_file_name_process(params_processor)
            system_params = params_processor.get_params_by_type(ParamType.SYSTEM)
            params_indicator = params_processor.get_params_by_type(ParamType.INDICATOR)
            poincare_method = params_processor.get_param("poincare_method")
            flow_map = create_system(system_params)
            path_fig = params_processor.get_param("path")
            filename = os.path.join(path_fig, fname + ".png")
            axis_data = params_processor.get_param("axis")

            params_indicator.update({"solver_method": flow_map.solver_method})
            logging.info("-- Ejecutando el cálculo la sección de poincaré.")
            t0_roots_list, values_roots_list = poincare_method.get_poincare_points_from_x0_grid(
                diff_system=flow_map.diff_system, **params_indicator
            )

            _ = plot_poincare_sections(
                values=values_roots_list,
                filename=filename,
                axis=axis_data,
            )
            logging.info(f"- Sección de Poincaré guardada en {filename}.")
            logging.info(f"- Fin del proceso de {self.name_dynamic_indicator}")
