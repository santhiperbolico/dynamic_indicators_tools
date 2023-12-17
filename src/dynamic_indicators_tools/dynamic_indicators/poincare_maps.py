from typing import Any, Callable, Dict, Sequence

import numpy as np
from scipy.optimize import root_scalar

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable


def poincare_section_generator(diff_var: DiffVariable, poincare_map):
    def init(t):
        return poincare_map(diff_var.solution(t))

    return init


def poincare_section_grid(
    diff_system: DiffSystem,
    poincare_map: Callable[[np.ndarray], float],
    solver_method: str,
    t_span: Sequence[float],
    x0: np.ndarray,
    n_points: int = 100,
    args: Sequence[Any] = None,
    params_solver: Dict[str, Any] = None,
    params_root: Dict[str, Any] = None,
) -> np.ndarray:
    if params_root is None:
        params_root = {"method": "bisect"}

    _, _ = diff_system.solve_function(solver_method, t_span, x0, args, params_solver)
    poincare_section = poincare_section_generator(diff_system.variable, poincare_map)
    t_values = np.linspace(t_span[0], t_span[1], n_points)
    t_roots = []
    for i in range(n_points - 1):
        root = None
        try:
            params_root.update({"bracket": [t_values[i], t_values[i + 1]]})
            root_result = root_scalar(poincare_section, **params_root)
            converged = root_result.converged
            root = root_result.root
        except ValueError:
            converged = False
        if converged:
            t_roots.append(root)
    t_roots = np.array(t_roots)
    return t_roots


def poincare_section_grid_ode(
    diff_system: DiffSystem,
    poincare_map: Callable[[np.ndarray], float],
    solver_method: str,
    t_span: Sequence[float],
    x0: np.ndarray,
    n_points: int = 100,
    args: Sequence[Any] = None,
    params_solver: Dict[str, Any] = None,
    params_root: Dict[str, Any] = None,
):
    if params_root is None:
        params_root = {"method": "bisect"}

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
    for i in range(t_values.size - 1):
        _, _ = diff_system.solve_function(
            solver_method, [t_values[i], t_values[i + 1]], x0_array[:, i], args, params_solver
        )
        poincare_section = poincare_section_generator(diff_system.variable, poincare_map)
        root = None
        params_root.update({"bracket": [t_values[i], t_values[i + 1]]})
        try:
            root_result = root_scalar(poincare_section, **params_root)
            converged = root_result.converged
            root = root_result.root
        except ValueError:
            converged = False
        if converged:
            t_roots.append(root)
    return np.array(t_roots)
