import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from attr import attrs
from scipy.optimize import root_scalar

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable

PoincareMapFunction = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]


def poincare_section_generator(
    diff_var: DiffVariable, poincare_map: PoincareMapFunction, pm_args: Sequence[Any] = None
) -> PoincareMapFunction:
    """
    Función que genera la función de la sección de poincaré compuesta de la interpolación
    de la solución de un sistema diferencial dado.

    Parameters
    ----------
    diff_var: DiffVariable
        Variable de un sistema diferencial que tiene almacenada la interpolación
        de la solución de dicho sistema diferencial.
    poincare_map: PoincareMapFunction
        Función que representa la sección de Poincaré, siendo está los puntos
        que anulan a la función.

    Returns
    -------
    init: PoincareMapFunction
        Función de poincaré compuesta con la solución x de un sistema diferencial
        De esta manera para un tiempo t devuelve el valor de x(t) de la función
        de Poincaré original.

    """
    if pm_args is None:
        pm_args = ()

    def init(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return poincare_map(diff_var.solution(t), *pm_args)

    return init


class PoincareSectionGrid(ABC):
    """
    Interfaz con la estructura de los métodos determinados
    a calcular las secciones de Poincaré de un sistema dado.
    """

    @staticmethod
    @abstractmethod
    def get_poincare_points(
        diff_system: DiffSystem,
        poincare_map: PoincareMapFunction,
        solver_method: str,
        t_span: Sequence[float],
        x0: np.ndarray,
        n_points: int = 100,
        args: Sequence[Any] = None,
        params_solver: Dict[str, Any] = None,
        params_root: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que calcula los puntos de tiempo y del espacio para la sección
        de poincaré indicada en el sistema indicado.

        Parameters
        ----------
        diff_system: DiffSystem
            Sistema diferencial del cual se quiere extraer los puntos de la sección de poinacaré.
        poincare_map: Callable[[np.ndarray], float],
            Función de la sección de Poincaré. La sección sería el conjunto de todos aquellos
            puntos que anulan a la función.
        solver_method: str
            Nombre del método para resolver DiffSystem. Ver DiffSystem.solve_function.
        t_span: Sequence[float]
            Intervalo de tiempo del cual se quiere extraer los puntos de Poincaré.
            Ver DiffSystem.solve_function
        x0: np.ndarray
            Condición inicial del sistema diferencial. Ver DiffSystem.solve_function
        n_points: int, default 100
            Número de puntos en el que se va a dividir el intervalo de tiempo para buscar
            los puntos de la solución en la sección de Poincaré.
        args: Sequence[Any], default None
            Argumentos asociados a la función del sistema diferencial.
            Ver DiffSystem.solve_function
        params_solver: Dict[str, Any], default None
            Parámetros del solver usado para resolver el sistema diferencial.
            Ver DiffSystem.solve_function
        params_root: Dict[str, Any], default None
            Parámetros del método para encontrar las raices de la función poincare_map

        Returns
        -------
        t_roots: np.ndarray
            Array con los valores de tiempo asociados a las intersecciones con la
            sección de Poincaré.
        values__roots: np.ndarray
            Valores de la solución en los tiempos de t_roots.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_poincare_points_from_x0_grid(
        x0_grid: Union[np.ndarray, List[np.ndarray]], **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Método que aplica el método get_poincare_points a una lista o array de condiciones
        iniciales x0_grid. Devuelve dos listas de longitud igual al tamaño de x0_grid con los
        resultados de cada condición inicial del método get_poincare_points.

        Parameters
        ----------
        x0_grid: Union[np.ndarray, List[np.ndarray]]
            Lista de condiciones iniciales. Si es un array debe de ser de dos dimensiones,
            con una forma (numero_condiciones_iniciales, dimension_condición_inicial).

        kwargs:
            Parámetros adicionales del método get_poincare_points

        Returns
        -------
        t0_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de t_roots de get_poincare_points para cada
            condición inicial.
        values_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de values_roots de get_poincare_points para cada
            condición inicial.
        """
        pass


@attrs
class PoincareSectionInterpolate(PoincareSectionGrid):
    name = "PoincareSectionInterpolate"

    @staticmethod
    def get_poincare_points(
        diff_system: DiffSystem,
        poincare_map: PoincareMapFunction,
        solver_method: str,
        t_span: Sequence[float],
        x0: np.ndarray,
        n_points: int = 100,
        args: Sequence[Any] = None,
        pm_args: Sequence[Any] = None,
        params_solver: Dict[str, Any] = None,
        params_root: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que calcula los puntos de tiempo y del espacio para la sección
        de poincaré indicada en el sistema indicado. Para ello resuelve el sistema
        diferencial e interpola la solución. Después, usando la función root_scalar
        de scipy.optimize en n_points-1 sub-intervalos de tiempo encuentra los
        valores de las raices de la función de la sección de Poincaré compuesta
        de la interpolación de la solución.

        Parameters
        ----------
        diff_system: DiffSystem
            Sistema diferencial del cual se quiere extraer los puntos de la sección de poinacaré.
        poincare_map: PoincareMapFunction,
            Función de la sección de Poincaré. La sección sería el conjunto de todos aquellos
            puntos que anulan a la función.
        solver_method: str
            Nombre del método para resolver DiffSystem. Ver DiffSystem.solve_function.
        t_span: Sequence[float]
            Intervalo de tiempo del cual se quiere extraer los puntos de Poincaré.
            Ver DiffSystem.solve_function
        x0: np.ndarray
            Condición inicial del sistema diferencial. Ver DiffSystem.solve_function
        n_points: int, default 100
            Número de puntos en el que se va a dividir el intervalo de tiempo para buscar
            los puntos de la solución en la sección de Poincaré.
        args: Sequence[Any], default None
            Argumentos asociados a la función del sistema diferencial.
            Ver DiffSystem.solve_function
        params_solver: Dict[str, Any], default None
            Parámetros del solver usado para resolver el sistema diferencial.
            Ver DiffSystem.solve_function
        params_root: Dict[str, Any], default None
            Parámetros del método para encontrar las raices de la función poincare_map.
            Ver scipy.optimize.root_scalar

        Returns
        -------
        t_roots: np.ndarray
            Array con los valores de tiempo asociados a las intersecciones con la
            sección de Poincaré.
        values__roots: np.ndarray
            Valores de la solución en los tiempos de t_roots.
        """
        default_params_root = {"method": "bisect"}
        if params_root is None:
            params_root = {}

        if params_solver is None:
            params_solver = {}

        default_params_root.update(params_root)
        params_general_solver = params_solver.copy()
        t_values = np.linspace(t_span[0], t_span[1], n_points)
        params_general_solver.update({"t_eval": t_values})
        _, _ = diff_system.solve_function(solver_method, t_span, x0, args, params_general_solver)
        poincare_section = poincare_section_generator(diff_system.variable, poincare_map, pm_args)

        t_roots = []
        mask_values = np.ones(t_values.size).astype(bool)
        sign_values = np.sign(poincare_section(t_values))
        mask_values[1:] = (sign_values[1:] + sign_values[:-1]) == 0
        t_values = t_values[mask_values]

        logging.info(f"Se han encontrado {t_values.size} puntos con cambios de signo.")

        for i in range(t_values.size - 1):
            root = None
            try:
                default_params_root.update({"bracket": [t_values[i], t_values[i + 1]]})
                root_result = root_scalar(poincare_section, **default_params_root)
                converged = root_result.converged
                root = root_result.root
            except ValueError:
                converged = False
            if converged:
                t_roots.append(root)
        t_roots = np.array(t_roots)
        values_roots = np.array([])
        if t_roots.size > 0:
            values_roots = diff_system.variable.solution(t_roots).T
        return t_roots, values_roots

    @staticmethod
    def get_poincare_points_from_x0_grid(
        x0_grid: Union[np.ndarray, List[np.ndarray]], **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Método que aplica el método get_poincare_points a una lista o array de condiciones
        iniciales x0_grid. Devuelve dos listas de longitud igual al tamaño de x0_grid con los
        resultados de cada condición inicial del método get_poincare_points.

        Parameters
        ----------
        x0_grid: Union[np.ndarray, List[np.ndarray]]
            Lista de condiciones iniciales. Si es un array debe de ser de dos dimensiones,
            con una forma (numero_condiciones_iniciales, dimension_condición_inicial).

        kwargs:
            Parámetros adicionales del método get_poincare_points

        Returns
        -------
        t0_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de t_roots de get_poincare_points para cada
            condición inicial.
        values_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de values_roots de get_poincare_points para cada
            condición inicial.
        """

        t0_roots_list = []
        values_roots_list = []
        if isinstance(x0_grid, list):
            x0_grid = np.array(x0_grid)

        for i in range(x0_grid.shape[0]):
            x0 = x0_grid[i, :]
            t0_roots, values_roots = PoincareSectionInterpolate.get_poincare_points(
                x0=x0, **kwargs
            )
            if t0_roots.size > 0:
                t0_roots_list.append(t0_roots)
                values_roots_list.append(values_roots)
        return t0_roots_list, values_roots_list


@attrs
class PoincareSectionOdeTimeRange(PoincareSectionGrid):
    name = "PoincareSectionOdeTimeRange"

    @staticmethod
    def get_poincare_points(
        diff_system: DiffSystem,
        poincare_map: PoincareMapFunction,
        solver_method: str,
        t_span: Sequence[float],
        x0: np.ndarray,
        n_points: int = 100,
        args: Sequence[Any] = None,
        pm_args: Sequence[Any] = None,
        params_solver: Dict[str, Any] = None,
        params_root: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que calcula los puntos de tiempo y del espacio para la sección
        de poincaré indicada en el sistema indicado. Para ello, resuelve el sistema diferencial
        sacando los valores para n_points de tiempo. Después, usando los valores para esos tiempos
        vuelve a resolver el sistema para cada uno de los n_points-1 subintervalos de tiempo,
        resolviendo el sistema, interpolando la solución y encontrando la raiz de la función
        de la sección de Poincaré compuesta de la interpolación de la solución del sistema
        diferencial.

        Parameters
        ----------
        diff_system: DiffSystem
            Sistema diferencial del cual se quiere extraer los puntos de la sección de poinacaré.
        poincare_map: PoincareMapFunction,
            Función de la sección de Poincaré. La sección sería el conjunto de todos aquellos
            puntos que anulan a la función.
        solver_method: str
            Nombre del método para resolver DiffSystem. Ver DiffSystem.solve_function.
        t_span: Sequence[float]
            Intervalo de tiempo del cual se quiere extraer los puntos de Poincaré.
            Ver DiffSystem.solve_function
        x0: np.ndarray
            Condición inicial del sistema diferencial. Ver DiffSystem.solve_function
        n_points: int, default 100
            Número de puntos en el que se va a dividir el intervalo de tiempo para buscar
            los puntos de la solución en la sección de Poincaré.
        args: Sequence[Any], default None
            Argumentos asociados a la función del sistema diferencial.
            Ver DiffSystem.solve_function
        params_solver: Dict[str, Any], default None
            Parámetros del solver usado para resolver el sistema diferencial.
            Ver DiffSystem.solve_function
        params_root: Dict[str, Any], default None
            Parámetros del método para encontrar las raices de la función poincare_map.
            Ver scipy.optimize.root_scalar

        Returns
        -------
        t_roots: np.ndarray
            Array con los valores de tiempo asociados a las intersecciones con la
            sección de Poincaré.
        values__roots: np.ndarray
            Valores de la solución en los tiempos de t_roots.
        """
        default_params_root = {"method": "bisect"}
        if params_root is None:
            params_root = {}
        default_params_root.update(params_root)

        if params_solver is None:
            params_solver = {}

        params_general_solver = params_solver.copy()
        if n_points:
            t_values = np.linspace(t_span[0], t_span[1], n_points)
            params_general_solver.update({"t_eval": t_values})
        t_values, x0_array = diff_system.solve_function(
            solver_method, t_span, x0, args, params_general_solver
        )
        t_roots = []
        values_roots = []
        poincare_section = poincare_section_generator(diff_system.variable, poincare_map, pm_args)
        mask_values = np.ones(t_values.size).astype(bool)
        sign_values = np.sign(poincare_section(t_values))
        mask_values[1:] = (sign_values[1:] + sign_values[:-1]) == 0
        t_values = t_values[mask_values]
        t_values[-1] = t_span[1]
        last_x0 = x0_array[-1, :].copy()
        x0_array = x0_array[mask_values, :]
        x0_array[-1, :] = last_x0

        logging.info(f"Se han encontrado {t_values.size} puntos con cambios de signo.")

        for i in range(t_values.size - 1):
            t_span_iteration = [float(t_values[i]), float(t_values[i + 1])]
            x0_iteration = x0_array[i, :]
            _, _ = diff_system.solve_function(
                solver_method=solver_method,
                t_span=t_span_iteration,
                x0=x0_iteration,
                args=args,
                params_solver=params_solver,
            )
            poincare_section = poincare_section_generator(
                diff_system.variable, poincare_map, pm_args
            )
            default_params_root.update({"bracket": [t_values[i], t_values[i + 1]]})
            try:
                root_result = root_scalar(poincare_section, **default_params_root)
                converged = root_result.converged
                root = root_result.root
            except ValueError:
                root = None
                converged = False
            if converged:
                values_roots.append(list(diff_system.variable.solution(root)))
                t_roots.append(root)
        t_roots = np.array(t_roots)
        values_roots = np.array(values_roots)
        return t_roots, values_roots

    @staticmethod
    def get_poincare_points_from_x0_grid(
        x0_grid: Union[np.ndarray, List[np.ndarray]], **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Método que aplica el método get_poincare_points a una lista o array de condiciones
        iniciales x0_grid. Devuelve dos listas de longitud igual al tamaño de x0_grid con los
        resultados de cada condición inicial del método get_poincare_points.

        Parameters
        ----------
        x0_grid: Union[np.ndarray, List[np.ndarray]]
            Lista de condiciones iniciales. Si es un array debe de ser de dos dimensiones,
            con una forma (numero_condiciones_iniciales, dimension_condición_inicial).

        kwargs:
            Parámetros adicionales del método get_poincare_points

        Returns
        -------
        t0_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de t_roots de get_poincare_points para cada
            condición inicial.
        values_roots_list: List[np.ndarray]
            Lista con los valores de los arrays de values_roots de get_poincare_points para cada
            condición inicial.
        """

        t0_roots_list = []
        values_roots_list = []
        if isinstance(x0_grid, list):
            x0_grid = np.array(x0_grid)

        for i in range(x0_grid.shape[0]):
            x0 = x0_grid[i, :]
            t0_roots, values_roots = PoincareSectionOdeTimeRange.get_poincare_points(
                x0=x0, **kwargs
            )
            if t0_roots.size > 0:
                t0_roots_list.append(t0_roots)
                values_roots_list.append(values_roots)
        return t0_roots_list, values_roots_list


def get_poincare_grid_method(method: str) -> PoincareSectionGrid:
    """
    Función que devuelve la clase PoincareSectionGrid asociada al método.

    Parameters
    ----------
    method: str
        Nombre de la clase del tipo PoincareSectionGrid que se va a utilizar.

    Returns
    -------
    get_poincare_points: PoincareSectionGrid
        Objeto de la clase asociada.
    """
    dic_methods = {
        PoincareSectionInterpolate.name: PoincareSectionInterpolate,
        PoincareSectionOdeTimeRange.name: PoincareSectionOdeTimeRange,
    }
    try:
        return dic_methods[method]()
    except KeyError:
        raise ValueError(
            f"El método de Poincaré {method} no está implementado."
            f"Usa alguno de {list(dic_methods.keys())}"
        )
