from typing import Any, Callable, Dict, NamedTuple, Sequence

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


class DoesntExisteSolverIntegerMethod(Exception):
    """
    Error que indica que la variable t la x no tiene la misma
    dimensión.
    """

    pass


DiferentialSystemFunction = Callable[[np.ndarray, np.ndarray, Sequence[Any]], np.ndarray]


class ParamsSolverFunction(NamedTuple):
    """
    Parámetros que tiene que tener una función que implementa
    un método integrador a un sistema diferencial.

    Parameters
    ----------
    function: DiferentialSystemFunction
        Función a resolver.
    t_span: Sequence[float]
        Secuencia de dos elementos t_0 y t
    x0: np.ndarray
        Array de dimensión (n_var, ) con las condiciones iniciales.
    args: Sequence[float], default None
        Parámetros de la función del sistema diferencial, ajenos de las
        variables del sistema.
    params_solver: Dict[str, Any]
        Diccionario con los parámetros del método integrador utulizado.
    """

    function: DiferentialSystemFunction
    t_span: Sequence[float]
    x0: np.ndarray
    args: Sequence[Any] = None
    params_solver: Dict[str, Any] = None


class ResultSolverFunction(NamedTuple):
    """
    Salida que tiene que tener una función que implementa
    un método integrador a un sistema diferencial.

    Returns
    ----------
    t_array: np.ndarray
            Vector con la malla de la variable t
    x_array: np.ndarray
        Array con los valores de las variables del sistema diferencial
    x_function: Callable[[np.ndarray], np.ndarray]
        Función interpoladora que dado un tiempo t nos devuelve x(t).
    """

    t_array: np.ndarray
    x_array: np.ndarray
    x_function: Callable[[np.ndarray], np.ndarray]


SolverIntegrators = Callable[[ParamsSolverFunction], ResultSolverFunction]


def _solve_function_ivp(params_solver_function: ParamsSolverFunction) -> ResultSolverFunction:
    """
    Función que implementa el método solve_ivp de scipy.integration
    para la resolución del sistema diferencial para las condiciones
    iniciales x0 en el intevalo t_span.

    Parameters
    ----------
    params_solver_function: ParamsSolverFunction
        Parámetros específicados en ParamsSolverFunction.
        El diccionari de params_solver hace referencia a los
        parámetros de solve_ivp de scipy.

    Returns
    -------
    result: ResultSolverFunction
        Resultado del integrador con la salida específicada en ResultSolver
    """
    t_span = params_solver_function.t_span
    x0 = params_solver_function.x0
    params_solver = params_solver_function.params_solver
    args = params_solver_function.args
    if params_solver is None:
        params_solver = {}
    if args is None:
        args = []
    method = params_solver.pop("method", "RK45")

    def fun_to_solve(t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return params_solver_function.function(t, x, *args)

    solution = solve_ivp(fun_to_solve, t_span, x0, method, dense_output=True, **params_solver)
    result = ResultSolverFunction(solution.t, solution.y.T, solution.sol)
    return result


def _solve_function_odeint(params_solver_function: ParamsSolverFunction) -> ResultSolverFunction:
    """
    Función que implementa el método odeint de scipy.integration
    para la resolución del sistema diferencial para las condiciones
    iniciales x0 en el intevalo t_span.

    Parameters
    ----------
    params_solver_function: ParamsSolverFunction
        Parámetros específicados en ParamsSolverFunction.
        El diccionario de params_solver hace referencia a los
        parámetros de odeint de scipy. Además se ha añadido:
            * n_grid_t: int, default 101
                Tamaño de la malla del array de t
            * kind_interpolate: str, default 'slinear'
                Método interpolador de interpd1 para
                calcular x_function de ResultSolverFunction

    Returns
    -------
    result: ResultSolverFunction
        Resultado del integrador con la salida específicada en ResultSolver
    """
    t_span = params_solver_function.t_span
    x0 = params_solver_function.x0
    params_solver = params_solver_function.params_solver
    args = params_solver_function.args
    if params_solver is None:
        params_solver = {}
    if args is None:
        args = []
    n_grid_t = params_solver.pop("n_grid_t", 101)
    kind_interpolate = params_solver.pop("kind_interpolate", "slinear")

    def fun_to_solve(x: np.ndarray, t: np.ndarray) -> list:
        result = params_solver_function.function(t, x, *args)
        return result.reshape(result.size)

    t_array = np.linspace(t_span[0], t_span[1], n_grid_t)
    x_array, _ = odeint(fun_to_solve, x0, t_array, full_output=True)

    def x_function(t: np.ndarray) -> np.ndarray:
        return interp1d(t_array, x_array, kind=kind_interpolate)(t).T

    result = ResultSolverFunction(t_array, x_array, x_function)
    return result


def get_solver_integrator(solver_method: str) -> SolverIntegrators:
    """
    Función que nos devuelve el integrador especificado.

    Parameters
    ----------
    solver_method: str
        Nombre del método integrador.

    Returns
    -------
    solver: SolverIntegrators
        Función integradora.
    """
    dic_solver_integrators = {"odeint": _solve_function_odeint, "solve_ivp": _solve_function_ivp}

    try:
        return dic_solver_integrators[solver_method]
    except KeyError:
        raise DoesntExisteSolverIntegerMethod(
            f"El método {solver_method} no está implementado."
            f"Utiliza alguno de {list(dic_solver_integrators.keys())}"
        )
