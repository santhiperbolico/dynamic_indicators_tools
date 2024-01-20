import logging
import os
from typing import Any, Dict, List

import numpy as np

from .ld_params import LD_PARAMS
from .ld_utils import lagrangian_descriptors
from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_process import (
    DynamicIndicator,
    create_system,
)
from dynamic_indicators_tools.dynamic_indicators.params_methods import (
    Param,
    ParamProcessor,
    ParamType,
)
from dynamic_indicators_tools.dynamic_indicators.plot_descriptors import plot_descriptors_map


class LagrangianDescriptor(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de los descriptores lagrangianos
    para una malla de puntos de condiciones iniciales.
    """

    name_dynamic_indicator: str = "lagrangian_descriptors"
    default_params: List[Param] = LD_PARAMS

    def create_params_processor(self, params: Dict[str, Any] = None) -> ParamProcessor:
        """
        Método que crea el params processor para el método de Lagrangian Descriptors.

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
        params_indicator.update(params.get(self.name_dynamic_indicator, {}))
        params_indicator.update(
            {
                "params_solver": {
                    "solver_method": params_indicator.get("solver_method", "solve_ivp"),
                    "t0": params_indicator.get("t0", 0),
                    "args": params_indicator.get("args_system", ()),
                }
            }
        )
        params_processor = ParamProcessor(self.default_params, params_indicator)
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
        n_xgrid = params_processor.get_param("n_xgrid")
        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid
        system_params = params_processor.get_params_by_type(ParamType.SYSTEM)
        params_indicator = params_processor.get_params_by_type(ParamType.INDICATOR)
        logging.info("-- Parámetros del proceso:")
        logging.info("\t T=%i" % params_indicator.get("t"))
        logging.info("\t Número de puntos =%i" % n_xgrid_total)
        logging.info("\t - Intervalo Tau usado = %s " % (str(params_indicator.get("tau"))))
        logging.info(
            "\t - Método de integración usado = %s " % params_indicator.get("method_integrate")
        )
        logging.info("\t Solver usado = %s " % system_params.get("solver_method"))
        logging.info(
            "\t Límite inferior de la malla = %s " % (str(list(params_indicator.get("x0_min"))))
        )
        logging.info(
            "\t Límite superior de la malla = %s " % (str(list(params_indicator.get("x0_max"))))
        )
        logging.info("\t Número máximo de hilos = %i" % params_indicator.get("n_jobs"))
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

        n_xgrid = params_processor.get_param("n_xgrid")
        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid
        system_name = params_processor.get_param("system_name")
        t = params_processor.get_param("t")
        fname = (
            f"{system_name}_{self.name_dynamic_indicator}_t_{t:.0f}_nx_grid_{n_xgrid_total:.0f}"
        )
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
            flow_map = create_system(system_params)
            path_fig = params_processor.get_param("path")
            filename = os.path.join(path_fig, fname + ".png")
            axis_data = params_processor.get_param("axis")
            log_scale = params_processor.get_param("log_scale_color")

            logging.info("-- Ejecutando el cálculo de malla.")
            grid_points, diff_f = lagrangian_descriptors(
                diff_system=flow_map.diff_system,
                opts_integrate={"args_func": flow_map.args_func},
                **params_indicator,
            )

            _, _ = plot_descriptors_map(
                grid_points[axis_data[0]],
                grid_points[axis_data[1]],
                values=diff_f,
                filename=filename,
                axis=axis_data,
                log_scale=log_scale,
            )
            logging.info(f"- Campo Lagrangian Descriptor guardado en {filename}.")
            logging.info(f"- Fin del proceso de {self.name_dynamic_indicator}")
