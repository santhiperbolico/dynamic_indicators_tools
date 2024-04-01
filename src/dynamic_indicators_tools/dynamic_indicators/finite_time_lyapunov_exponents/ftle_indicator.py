import logging
import os
from typing import Any, Dict, List

import numpy as np
from attr import attrs

from .ftle_params import FTLE_ELEMENT_WISE_PARAMS, FTLE_GRID_PARAMS, FTLE_VARIATIONAL_EQUATIONS
from .ftle_utils import ftl_variational_equations, ftle_element_wise, ftle_grid
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


@attrs
class FtleElementWise(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de los exponentes de
    Lyapunov en tiempos finitos calculado elemento a elemento.
    """

    name_dynamic_indicator: str = "ftle_element_wise"
    default_params: List[Param] = FTLE_ELEMENT_WISE_PARAMS

    def create_params_processor(self, params: Dict[str, Any] = None) -> ParamProcessor:
        """
        Método que crea el params processor para el método de FtleElementWise.

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
        params_processor = ParamProcessor(self.default_params, params_indicator)
        h_steps = params_processor.get_param("h_steps")
        n_xgrid = params_processor.get_param("n_xgrid")

        if h_steps is None:
            x_max = params_processor.get_param("x0_max")
            x_min = params_processor.get_param("x0_min")
            params_processor.update_param("h_steps", (x_max - x_min) / (n_xgrid - 1))
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
        logging.info("\t Paso de la integración = %s " % (str(params_indicator.get("h_steps"))))
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
        h_steps = params_processor.get_param("h_steps")
        t = params_processor.get_param("t")
        fname = (
            f"{system_name}_{self.name_dynamic_indicator}_t_{t:.0f}_nx_grid_{n_xgrid_total:.0f}"
        )
        if isinstance(h_steps, np.ndarray):
            logging.info("\t -Paso de la derivada = %.4f" % h_steps.max())
            fname = f"{fname}_h_{h_steps.max():.4f}"
        if isinstance(h_steps, float):
            logging.info("\t -Paso de la derivada = %.4f" % h_steps)
            fname = f"{fname}_h_{h_steps:.4f}"
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
            system_params = params_processor.get_params_by_type(ParamType.SYSTEM)
            params_indicator = params_processor.get_params_by_type(ParamType.INDICATOR)
            flow_map = create_system(system_params)
            fname = self.create_file_name_process(params_processor)
            path_fig = params_processor.get_param("path")
            filename = os.path.join(path_fig, fname + ".png")
            axis_data = params_processor.get_param("axis")
            logging.info("-- Ejecutando el cálculo punto a punto.")
            grid_points, zz = ftle_element_wise(flow_map, **params_indicator)
            _, _ = plot_descriptors_map(
                grid_points[axis_data[0]],
                grid_points[axis_data[1]],
                values=zz,
                filename=filename,
                axis=axis_data,
            )
            logging.info(f"- Campo FTLE guardado en {filename}.")
            logging.info(f"- Fin del proceso de {self.name_dynamic_indicator}")


@attrs
class FtleGrid(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de los exponentes de
    Lyapunov en tiempos finitos calculado en malla.
    """

    name_dynamic_indicator: str = "ftle_grid"
    default_params: List[Param] = FTLE_GRID_PARAMS

    def create_params_processor(self, params: Dict[str, Any] = None) -> ParamProcessor:
        """
        Método que crea el params processor para el método de FtleElementWise.

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
            logging.info("-- Ejecutando el cálculo de malla.")
            grid_points, diff_f = ftle_grid(flow_map, **params_indicator)
            _, _ = plot_descriptors_map(
                grid_points[axis_data[0]],
                grid_points[axis_data[1]],
                values=diff_f,
                filename=filename,
                axis=axis_data,
            )
            logging.info(f"- Campo FTLE guardado en {filename}.")
            logging.info(f"- Fin del proceso de {self.name_dynamic_indicator}")


@attrs
class FtleVariationalEquations(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de los exponentes de
    Lyapunov en tiempos finitos calculado punto a punto a través
    de las ecuaciones diferenciales variacionales.
    """

    name_dynamic_indicator: str = "ftle_variational_equations"
    default_params: List[Param] = FTLE_VARIATIONAL_EQUATIONS

    def create_params_processor(self, params: Dict[str, Any] = None) -> ParamProcessor:
        """
        Método que crea el params processor para el método de FtleElementWise.

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
        if "var_system" in params_indicator.keys():
            params_indicator.update({"function": params_indicator.get("var_system")})
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
        indicador en colores. En vez de calcular la jacobiana numéricamente la incluye
        en el sistema diferencial en forma de ecuaciones variacionales.

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

            logging.info("-- Ejecutando el cálculo de malla con ecuaciones variacionales.")
            grid_points, diff_f = ftl_variational_equations(flow_map, **params_indicator)
            _, _ = plot_descriptors_map(
                grid_points[axis_data[0]],
                grid_points[axis_data[1]],
                values=diff_f,
                filename=filename,
                axis=axis_data,
            )
            logging.info(f"- Campo FTLE guardado en {filename}.")
            logging.info(f"- Fin del proceso de {self.name_dynamic_indicator}")
