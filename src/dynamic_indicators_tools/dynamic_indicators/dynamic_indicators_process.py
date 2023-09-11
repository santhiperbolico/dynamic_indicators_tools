import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict
from warnings import warn

import numpy as np
from attr import attrs

from dynamic_indicators_tools.config_files.get_params_config import format_symbolic_number
from dynamic_indicators_tools.differentials_systems.diff_system import (
    DiffSystem,
    DiffVariable,
    FlowMap,
)
from dynamic_indicators_tools.dynamic_indicators.finite_time_lyapunov_exponents import (
    ftl_variational_equations,
    ftle_element_wise,
    ftle_grid,
)
from dynamic_indicators_tools.dynamic_indicators.lagrangian_descriptors import (
    lagrangian_descriptors,
)
from dynamic_indicators_tools.dynamic_indicators.plot_descriptors import (
    plot_descriptors_map,
    plot_extremals_solutions,
)


def create_system(system: Dict[str, Any], system_params: Dict[str, Any]) -> FlowMap:
    """
    Función que devueve el FlowMap asociado a la función de systems.

    Parameters
    ----------
    system: Dict[str, Any]
        Diccionario con las funciones del sistema diferencial
    system_params: Dict[str, Any]
        Diccionario con los parámetros generales de los indicadores y del sistema.

    Returns
    -------
    flow_map: FlowMap
        FlowMap asociado al sistema diferencial que define systems.
    """
    function_system = system["function"]
    system_name = system_params.get("system_name")
    args_system = system_params.get("args_system")
    t0 = system_params.get("t0")
    solver_method = system_params.get("solver_method")

    logging.info("\t - Creando el sistema %s" % system_name)
    x = DiffVariable("x")
    flow_map = FlowMap(diff_system=DiffSystem(x, function_system), t0=t0)
    flow_map.set_params_fun_solver(solver_method=solver_method, args_func=args_system)
    return flow_map


class DynamicIndicatorNotExist(Exception):
    pass


@attrs
class DynamicIndicator(ABC):
    @abstractmethod
    def process(
        self,
        system: Dict[str, Any],
        system_params: Dict[str, Any],
        dynamic_indicators_params: Dict[str, Any],
    ) -> None:
        """
        Proceso que ejecuta el indicador dinámico
        """
        pass


@attrs
class FtleElementWise(DynamicIndicator):
    """
    Proceso que recoge el indicador dinámico de los exponentes de
    Lyapunov en tiempos finitos calculado elemento a elemento.
    """

    def process(
        self,
        system: Dict[str, Any],
        system_params: Dict[str, Any],
        dynamic_indicators_params: Dict[str, Any],
    ) -> None:
        """
        Proceso que ejecuta el indicador dinámico para una malla de puntos sacando
        una gráfica en el plano indicado dentro de system_params con los valores del
        indicador en colores. Guarda las gráficas en la ruta especificada.

        Parameters
        ----------
        system: Dict[str, Any]
            Diccionario con las funciones del sistema diferencial
        system_params: Dict[str, Any]
            Diccionario con los parámetros generales de los indicadores y del sistema.
        dynamic_indicators_params: Dict[str, Any]
            Diccionario con los parámetros del indicador. Debe aparecer un parámetro
            "execute" que indica si se quiere procesar dicho indicador, por defecto lo
            toma como False.
        """
        logging.info("-- Cargando parámetros.")
        path_fig = system_params.get("path", ".")
        axis_data = system_params["axis"]
        system_name = system_params.get("system_name")
        t1 = format_symbolic_number(system_params.get("t1"))
        n_xgrid = system_params.get("n_xgrid")
        x_min = system_params.get("x_min")
        x_max = system_params.get("x_max")
        n_jobs = system_params.get("n_jobs")
        projection_config = system_params.get("projection_config")
        flow_map = create_system(system, system_params)
        h_steps = dynamic_indicators_params.get("h_steps")
        params_time_close = dynamic_indicators_params.get("params_t_close")

        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid
        if h_steps is None:
            h_steps = (x_max - x_min) / (n_xgrid - 1)
        logging.info("-- Parámetros del proceso:")
        logging.info("\t T=%i" % t1)
        logging.info("\t Número de puntos =%i" % n_xgrid_total)
        logging.info("\t Paso de la integración = %s " % (str(h_steps)))
        logging.info("\t Solver usado = %s " % flow_map.solver_method)
        logging.info("\t Límite inferior de la malla = %s " % (str(list(x_min))))
        logging.info("\t Límite superior de la malla = %s " % (str(list(x_max))))
        logging.info("\t Número máximo de hilos = %i" % n_jobs)

        logging.info("-- Ejecutando el cálculo punto a punto.")
        fname = f"{system_name}_element_wise_t_{t1:.0f}_nx_grid_{n_xgrid_total:.0f}"
        if isinstance(h_steps, np.ndarray):
            logging.info("\t -Paso de la derivada = %.4f" % h_steps.max())
            fname = f"{fname}_h_{h_steps.max():.4f}"
        if isinstance(h_steps, float):
            logging.info("\t -Paso de la derivada = %.4f" % h_steps)
            fname = f"{fname}_h_{h_steps:.4f}"
        filename = os.path.join(path_fig, fname + ".png")
        grid_points, zz = ftle_element_wise(
            flow_map,
            t1,
            x_min,
            x_max,
            n_xgrid,
            h_steps=h_steps,
            n_jobs=n_jobs,
            params_t_close=params_time_close,
            projection_config=projection_config,
        )
        _, _ = plot_descriptors_map(
            grid_points[axis_data[0]],
            grid_points[axis_data[1]],
            values=zz,
            filename=filename,
            axis=axis_data,
        )


@attrs
class FtleGrid(DynamicIndicator):
    def process(
        self,
        system: Dict[str, Any],
        system_params: Dict[str, Any],
        dynamic_indicators_params: Dict[str, Any],
    ) -> None:
        """
        Proceso que ejecuta el indicador dinámico para una malla de puntos sacando
        una gráfica en el plano indicado dentro de system_params con los valores del
        indicador en colores. Guarda las gráficas en la ruta especificada.

        Parameters
        ----------
        system: Dict[str, Any]
            Diccionario con las funciones del sistema diferencial
        system_params: Dict[str, Any]
            Diccionario con los parámetros generales de los indicadores y del sistema.
        dynamic_indicators_params: Dict[str, Any]
            Diccionario con los parámetros del indicador. Debe aparecer un parámetro
            "execute" que indica si se quiere procesar dicho indicador, por defecto lo
            toma como False.
        """
        logging.info("-- Cargando parámetros.")
        path_fig = system_params.get("path", ".")
        axis_data = system_params["axis"]
        system_name = system_params.get("system_name")
        t1 = format_symbolic_number(system_params.get("t1"))
        n_xgrid = system_params.get("n_xgrid")
        x_min = system_params.get("x_min")
        x_max = system_params.get("x_max")
        n_jobs = system_params.get("n_jobs")
        projection_config = system_params.get("projection_config")
        flow_map = create_system(system, system_params)
        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid

        logging.info("-- Parámetros del proceso:")
        logging.info("\t T=%i" % t1)
        logging.info("\t Número de puntos =%i" % n_xgrid_total)
        logging.info("\t Solver usado = %s " % flow_map.solver_method)
        logging.info("\t Límite inferior de la malla = %s " % (str(list(x_min))))
        logging.info("\t Límite superior de la malla = %s " % (str(list(x_max))))
        logging.info("\t Número máximo de hilos = %i" % n_jobs)

        logging.info("-- Ejecutando el cálculo de malla.")
        grid_points, diff_f = ftle_grid(
            flow_map, t1, x_min, x_max, n_xgrid, n_jobs, projection_config=projection_config
        )
        fname = f"{system_name}_grid_t_{t1:.0f}_nx_grid_{n_xgrid_total:.0f}.png"
        filename = os.path.join(path_fig, fname)
        _, _ = plot_descriptors_map(
            grid_points[axis_data[0]],
            grid_points[axis_data[1]],
            values=diff_f,
            filename=filename,
            axis=axis_data,
        )


@attrs
class FtleVariationalEquations(DynamicIndicator):
    def process(
        self,
        system: Dict[str, Any],
        system_params: Dict[str, Any],
        dynamic_indicators_params: Dict[str, Any],
    ) -> None:
        """
        Proceso que ejecuta el indicador dinámico para una malla de puntos sacando
        una gráfica en el plano indicado dentro de system_params con los valores del
        indicador en colores. En vez de calcular la jacobiana numéricamente la incluye
        en el sistema diferencial en forma de ecuaciones variacionales.

        Parameters
        ----------
        system: Dict[str, Any]
            Diccionario con las funciones del sistema diferencial
        system_params: Dict[str, Any]
            Diccionario con los parámetros generales de los indicadores y del sistema.
        dynamic_indicators_params: Dict[str, Any]
            Diccionario con los parámetros del indicador. Debe aparecer un parámetro
            "execute" que indica si se quiere procesar dicho indicador, por defecto lo
            toma como False.
        """
        logging.info("-- Cargando parámetros.")
        path_fig = system_params.get("path", ".")
        axis_data = system_params["axis"]
        system_name = system_params.get("system_name")
        t1 = format_symbolic_number(system_params.get("t1"))
        n_xgrid = system_params.get("n_xgrid")
        x_min = system_params.get("x_min")
        x_max = system_params.get("x_max")
        n_jobs = system_params.get("n_jobs")
        projection_config = system_params.get("projection_config")
        system_ve = system.copy()
        system_ve["function"] = dynamic_indicators_params.get("system")
        flow_map = create_system(system_ve, system_params)
        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid

        logging.info("-- Parámetros del proceso:")
        logging.info("\t T=%i" % t1)
        logging.info("\t Número de puntos =%i" % n_xgrid_total)
        logging.info("\t Solver usado = %s " % flow_map.solver_method)
        logging.info("\t Límite inferior de la malla = %s " % (str(list(x_min))))
        logging.info("\t Límite superior de la malla = %s " % (str(list(x_max))))
        logging.info("\t Número máximo de hilos = %i" % n_jobs)

        logging.info("-- Ejecutando el cálculo de malla con ecuaciones variacionales.")
        grid_points, diff_f = ftl_variational_equations(
            flow_map, t1, x_min, x_max, n_xgrid, n_jobs, projection_config=projection_config
        )
        fname = f"{system_name}_variational_equations_t_{t1:.0f}_nx_grid_{n_xgrid_total:.0f}.png"
        filename = os.path.join(path_fig, fname)
        _, _ = plot_descriptors_map(
            grid_points[axis_data[0]],
            grid_points[axis_data[1]],
            values=diff_f,
            filename=filename,
            axis=axis_data,
        )


@attrs
class LagrangianDescriptor(DynamicIndicator):
    def process(
        self,
        system: Dict[str, Any],
        system_params: Dict[str, Any],
        dynamic_indicators_params: Dict[str, Any],
    ):
        """
        Proceso que ejecuta el indicador dinámico para una malla de puntos sacando
        una gráfica en el plano indicado dentro de system_params con los valores del
        indicador en colores. Guarda las gráficas en la ruta especificada.

        Parameters
        ----------
        system: Dict[str, Any]
            Diccionario con las funciones del sistema diferencial
        system_params: Dict[str, Any]
            Diccionario con los parámetros generales de los indicadores y del sistema.
        dynamic_indicators_params: Dict[str, Any]
            Diccionario con los parámetros del indicador. Debe aparecer un parámetro
            "execute" que indica si se quiere procesar dicho indicador, por defecto lo
            toma como False.
        """
        logging.info("-- Cargando parámetros.")
        path_fig = system_params.get("path", ".")
        axis_data = system_params["axis"]
        system_name = system_params.get("system_name")
        t1 = format_symbolic_number(system_params.get("t1"))
        n_xgrid = system_params.get("n_xgrid")
        x_min = system_params.get("x_min")
        x_max = system_params.get("x_max")
        n_jobs = system_params.get("n_jobs")
        projection_config = system_params.get("projection_config")
        flow_map = create_system(system, system_params)
        n_xgrid_total = np.prod(n_xgrid) if isinstance(n_xgrid, np.ndarray) else n_xgrid

        tau = dynamic_indicators_params.get("tau")
        method_integrate = dynamic_indicators_params.get("method_integrate")
        log_scale = dynamic_indicators_params.get("log_scale_color", False)
        plot_orbits = dynamic_indicators_params.get("plot_orbits")
        params_solver = {
            "solver_method": flow_map.solver_method,
            "t0": flow_map.t0,
            "args": flow_map.args_func,
        }

        logging.info("-- Parámetros del proceso:")
        logging.info("\t - T=%i" % t1)
        logging.info("\t - Número de puntos =%i" % n_xgrid_total)
        logging.info("\t - Intervalo Tau usado = %s " % (str(tau)))
        logging.info("\t - Método de integración usado = %s " % method_integrate)
        logging.info("\t - Solver usado = %s " % flow_map.solver_method)
        logging.info("\t - Límite inferior de la malla = %s " % (str(list(x_min))))
        logging.info("\t - Límite superior de la malla = %s " % (str(list(x_max))))
        logging.info("\t - Número máximo de hilos = %i" % n_jobs)

        logging.info("-- Ejecutando descriptores Lagrangianos punto a punto.")
        grid_points, zz = lagrangian_descriptors(
            flow_map.diff_system,
            t1,
            x_min,
            x_max,
            n_xgrid,
            tau=tau,
            method_integrate=method_integrate,
            n_jobs=n_jobs,
            opts_integrate={"args_func": flow_map.args_func},
            projection_config=projection_config,
            params_solver=params_solver,
        )

        fname = f"{system_name}_lag_desc_t_{t1:.0f}_nx_grid_{n_xgrid_total:.0f}.png"
        filename = os.path.join(path_fig, fname)
        fig, ax = plot_descriptors_map(
            grid_points[axis_data[0]],
            grid_points[axis_data[1]],
            values=zz,
            filename=filename,
            axis=axis_data,
            log_scale=log_scale,
        )
        if plot_orbits:
            _ = plot_extremals_solutions(
                ax=ax,
                extremals_functionals=system["extremals"],
                args_func=flow_map.args_func,
                x_min=x_min,
                x_max=x_max,
                t1=t1,
                diff_system=flow_map.diff_system,
                solver_method=flow_map.solver_method,
            )
            fig.savefig(filename)


DynamicIndicatorDict = {
    "ftle_element_wise": FtleElementWise,
    "ftle_grid": FtleGrid,
    "lagrangian_descriptors": LagrangianDescriptor,
    "ftle_variational_equations": FtleVariationalEquations,
}


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


def main_process_di(
    system: Dict[str, Any], system_params: Dict[str, Any], dynamic_indicators: Dict[str, Any]
) -> None:
    """
    Función que crea las gráficas de indicadores dinámicos para el problema
    configurado a través de los parámetros system y system_params. Cada indicador dinámico
    está especificado en dynamic_indicators donde la key es el nombre y los valores
    un diccionario con los parámetros del indicador. En ese diccionario debe aparecer
    un parámetro "execute" que indica si se quiere procesar dicho indicador, por defecto
    lo toma como False.

    Parameters
    ----------
    system: Dict[str, Any]
        Diccionario con las funciones del sistema diferencial
    system_params: Dict[str, Any]
        Diccionario con los parámetros generales de los indicadores y del sistema.
    dynamic_indicators: Dict[str, Any]
        Diccionario donde Cada indicador dinámico está especificado con la key como el nombre
        y los valores como un diccionario con los parámetros del indicador. En ese diccionario
        debe aparecer un parámetro "execute" que indica si se quiere procesar dicho indicador,
        por defecto lo toma como False.
    """
    for dynamic_indicator_name, dynamic_indicator_params in dynamic_indicators.items():
        try:
            dynamic_indicator_object = get_dynamic_indicator(dynamic_indicator_name)
        except DynamicIndicatorNotExist:
            warn(
                f"El indicador {dynamic_indicator_name} no existe. "
                "Se continua con el siguiente indicador."
            )
            continue
        if dynamic_indicator_params.pop("execute", False):
            logging.info(f"Ejecutando {dynamic_indicator_name}")
            dynamic_indicator_object.process(system, system_params, dynamic_indicator_params)
            logging.info(f"- Fin del proceso de {dynamic_indicator_name}")
            logging.info("----------------------------------------------")
