from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from attr import attrib, attrs
from tqdm import tqdm

from dynamic_indicators_tools.differentials_systems.data_transformations import (
    Projection,
    project_grid_data,
)
from dynamic_indicators_tools.differentials_systems.solver_integrators import (
    DiferentialSystemFunction,
    ParamsSolverFunction,
    get_solver_integrator,
)

VariableFunction = Callable[[np.ndarray], np.ndarray]


class DoesntExistSolution(Exception):
    """
    Error que indica que no existe solución implementada.
    """

    pass


class DoesntCoincideDimension(Exception):
    """
    Error que indica que la variable t la x no tiene la misma
    dimensión.
    """

    pass


@attrs
class DiffVariable:
    """
    Clase que recoge los objetos de variables utilizadas
    en ecuaciones diferenciales.
    """

    name = attrib(type=str, init=True)
    values = attrib(type=np.ndarray, init=True, default=None)
    solution = attrib(type=VariableFunction, init=False, default=None)

    def __call__(self, t: Union[int, float, np.ndarray]) -> np.ndarray:
        if self.solution:
            return self.solution(t)
        raise DoesntExistSolution("No esta instanciada la solución de la variable.")

    def set_solution(self, sol_func: VariableFunction) -> None:
        self.solution = sol_func

    def set_values(self, t: np.ndarray, values: np.ndarray) -> None:
        if len(t.shape) < 2:
            t = t.reshape(-1, 1)
        if len(values.shape) < 2:
            values = values.reshape(1, -1)
        if t.shape[0] != values.shape[0]:
            raise DoesntCoincideDimension(
                f"El tamaño los valores de la variable {self.name} no "
                f"coincide con el tamaño de la malla t."
            )
        self.values = np.concatenate((t, values), axis=1)


EquationFunction = Callable[[DiffVariable], np.ndarray]


@attrs
class DiffSystem:
    """
    Clase que recoge las funcionalidades básicas de los sistemas diferenciales,
    incluyendo los integradores.
    """

    variable = attrib(type=DiffVariable, init=True)
    function = attrib(type=EquationFunction, init=True)

    def get_fun_to_solve(self) -> DiferentialSystemFunction:
        """
        Método que genera las funciones del sistema diferencia
        adaptada al método del integrador.

        Returns
        -------
        fun_to_solve: DiferentialSystemFunction
            Función generada.
        """

        def fun_to_solve(t: np.ndarray, x: np.ndarray, *args: Sequence[Any]) -> np.ndarray:
            xv = DiffVariable("xv")
            if isinstance(t, (float, int)):
                xv.set_values(np.array([[t]]), x)
            if isinstance(t, np.ndarray):
                xv.set_values(t, x)
            return self.function(xv, *args)

        return fun_to_solve

    def solve_function(
        self,
        solver_method: str,
        t_span: Sequence[float],
        x0: np.ndarray,
        args: Sequence[Any] = None,
        params_solver: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método que implementa el integrador solver_method
        implementados en esta clase.

        Parameters
        ----------
        solver_method: str
            Indica si se utiliza el método odeint o solve_ivp de
            scipy.integrate
        t_span: Sequence[float]
            Secuencia de dos elementos t_0 y t
        x0: np.ndarray
            Array de dimensión (n_var, ) con las condiciones iniciales.
        args: Sequence[float], default None
            Parámetros de la función del sistema diferencial, ajenos de las
            variables del sistema.
        params_solver

        Returns
        -------
        t_array: np.ndarray
            Vector con la malla de la variable t
        x_array: np.ndarray
            Array con los valores de las variables del sistema diferencial
        """
        function_to_solve = self.get_fun_to_solve()
        params_solver_function = ParamsSolverFunction(
            function=function_to_solve,
            t_span=t_span,
            x0=x0,
            args=args,
            params_solver=params_solver,
        )
        result = get_solver_integrator(solver_method)(params_solver_function)
        self.variable.set_solution(result.x_function)
        self.variable.set_values(result.t_array, result.x_array)
        return result.t_array, result.x_array


@attrs
class FlowMap:
    """
    Clase que recoge los métodos asociados al cálculo de
    flujos y de FTLE.
    """

    diff_system = attrib(type=DiffSystem, init=True)
    t0 = attrib(type=Union[int, float], init=True, default=0)
    solver_method = attrib(type=str, init=False, default="solve_ivp")
    args_func = attrib(type=Sequence[Any], init=False, default=None)
    params_solver = attrib(type=Dict[str, Any], init=False, default=None)

    def __call__(
        self,
        t: Union[int, float],
        x0: np.ndarray,
    ) -> np.ndarray:
        """
        Función que devuelve el flujo del sistema diferencial asociado
        para las condiciones iniciales x0 en el instante t.

        Parameters
        ----------
        t: Union[int, float]
            Tiempo
        x0: np.ndarray
            Array (N, n_variables) de condiciones iniciales.

        Returns
        -------
        flow: np.ndarray
            Solución del flujo.
        """
        if len(x0.shape) < 2:
            x0 = x0.reshape(1, -1)

        t_span = [self.t0, t]
        flow = np.zeros(x0.shape)
        solver_method, args, params_solver = self.get_params_fun_solver()
        for i in range(x0.shape[0]):
            _, flow_x = self.diff_system.solve_function(
                solver_method, t_span, x0[i, :], args, params_solver
            )
            flow[i, :] = flow_x[-1, :]
        return flow

    def get_time_close(
        self,
        t: Union[int, float],
        x0: np.ndarray,
        time_delta: float,
        dimensions_close: List[bool] = None,
        mod_solution: np.ndarray = None,
    ) -> float:
        """
        Función que calcula el instante T dentro de [t(1-time_delta), t(1+time_delta)]
        donde el valor del flujo sea lo más cercano a x0. Los parámetros
        dimensions_close determina que variables usamos para medir la distancia a x0
        y mod_solution indica el módulo que usamos para calcular la distancia.

        Parameters
        ----------
        t: Union[int, float]
            Tiempo
        x0: np.ndarray
            Array (N, n_variables) de condiciones iniciales.
        time_delta: float
            Valor entre 0 y 1 que determina el tamaño del intervalo de búsqueda.
            [t(1-time_delta), t(1+time_delta)]
        dimensions_close: List[bool], default None
            Lista de booleanos que determina las dimensiones usadas para calcular
            la distancia con x0. Por deecto se utilizan todas.
        mod_solution: np.ndarray, default None
            Array que indica, para cada dimensión usada en el cálculo de la distancia
            con x0, el módulo. Por ejemplo, en el caso del péndulo simple el módulo
            usado para theta sería 2pi. En el caso de no indicarlo no se aplica.

        Returns
        -------
        t_close: np.ndarray
            Solución del flujo.
        """
        if len(x0.shape) < 2:
            x0 = x0.reshape(1, -1)

        t_span = [self.t0, t * (1 + time_delta)]
        solver_method, args, params_solver = self.get_params_fun_solver()
        t_array_close = np.linspace(t * (1 - time_delta), t * (1 + time_delta), 500)
        _, _ = self.diff_system.solve_function(
            solver_method, t_span, x0[0, :], args, params_solver
        )
        flow_close = self.diff_system.variable(t_array_close).T
        if dimensions_close is None:
            dimensions_close = [True] * x0.shape[1]
        distance = np.linalg.norm(flow_close[:, dimensions_close] - x0[0, dimensions_close])
        if isinstance(mod_solution, np.ndarray):
            distance = np.linalg.norm(
                flow_close[:, dimensions_close] % mod_solution
                - x0[0, dimensions_close] % mod_solution,
                axis=1,
            )
        t_close = t_array_close[distance.argmin()]
        return t_close

    def set_params_fun_solver(
        self,
        solver_method: str = None,
        args_func: Sequence[Any] = None,
        params_solver: Dict[str, Any] = None,
    ) -> None:
        """
        Método que modifica los valores de los parámetros
        solver_method, args_func y params_solver

        Parameters
        ----------
        solver_method: str, default None
            Nombre del método solver.
        args_func: Sequence[Any]
            Secuencia con los parámetros del sistema diferencial
        params_solver: Dict[str, Any], default None
            Diccionario con los parámetros del solver.
        """
        self.solver_method = solver_method or self.solver_method
        self.args_func = args_func or self.args_func
        self.params_solver = params_solver or self.params_solver

    def get_params_fun_solver(self) -> Tuple[str, Sequence[Any], Dict[str, Any]]:
        """
        Método que devuelve los parámetos de solver_method, args_func y params_solver.

        Returns
        -------
        solver_method: str, default None
            Nombre del método solver.
        args_func: Sequence[Any]
            Secuencia con los parámetros del sistema diferencial
        params_solver: Dict[str, Any], default None
            Diccionario con los parámetros del solver.
        """
        return self.solver_method, self.args_func, self.params_solver

    def flow_grid(
        self,
        t: Union[int, float],
        x0_min: np.ndarray,
        x0_max: np.ndarray,
        n_xgrid: Union[int, np.ndarray],
        n_jobs: int = 1,
        projection_config: Dict[int, Projection] = None,
    ) -> Tuple[Union[Sequence[np.ndarray], str], np.ndarray]:
        """
        Método que crea una malla de condiciones iniciales, donde los
        limítes están definidos por x0_min_grid y x0_max_grid y la cantidad
        de puntos son nx_grid^(numero de variables), y calcula el valor del
        flujo para estos puntos.

        Parameters
        ----------
        t: Union[int, float]
            Valor del tiempo.
        x0_min: np.ndarray
            Array de dimensión (n_variables,) que indica el valor inferior
            de la malla.
        x0_max: np.ndarray
            Array de dimensión (n_variables,) que indica el valor superior
            de la malla.
        n_xgrid: Union[int, np.ndarray]
            Número de puntos generados por variable, donde el número
            total de puntos será nx_grid^(n_variables.). Si pasamos
            un entero se aplicará a todas las dimensiones. Si es una
            secuencia se aplicará según el orden de las dimensiones.
        n_jobs: int, default 1
            Número máximo de jobs en la paralelización.
        projection_config: Dict[int, Projection], default None
            Diccionario que recoge las projecciones que se hagan en cada dimensión.

        Returns
        -------
        grid_points: Sequence[np.ndarray]
            Lista con la malla para cada dimensión. Cada elemento
            es n array de dimensión (nx_grid,..., <n_variables>, ...,nx_grid).
        zz: np.ndarray:
            Array con el valor del flujo para cada punto de la malla. Este
            array es de dimensión (nx_grid,..., <n_variables>, ...,nx_grid, n_variables).
        """
        n_var = x0_max.size
        if isinstance(n_xgrid, int):
            n_xgrid = np.ones(n_var).astype(int) * n_xgrid
        if x0_min.shape != x0_max.shape:
            raise DoesntCoincideDimension(
                "La dimensión de x0_min_grid y x0_max_grid" " deben ser iguales."
            )
        if n_xgrid.size != n_var:
            raise DoesntCoincideDimension(
                "La dimensión de nx_grid no coincide con el número de variables."
            )
        grid_points = np.meshgrid(
            *[np.linspace(x0_min[i], x0_max[i], n_xgrid[i]) for i in range(n_var)]
        )
        grid_points = project_grid_data(grid_points, projection_config)
        flow = np.zeros(grid_points[0].shape + (n_var,))
        it = np.nditer(grid_points[0], flags=["multi_index"])

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            multi_index_list: List[Sequence[int]] = []
            index = 0
            while not it.finished:
                x0 = np.array([gp[it.multi_index] for gp in grid_points])
                if np.isnan(x0).any():
                    future = executor.submit(lambda x: np.nan, x0)
                else:
                    future = executor.submit(self, t, x0)
                futures.append(future)
                multi_index_list.append(it.multi_index)
                index += 1
                _ = it.iternext()
            for i, future in enumerate(tqdm(futures)):
                f_x0 = future.result()
                flow[multi_index_list[i]] = f_x0
        return grid_points, flow
