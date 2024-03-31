from typing import Any, Callable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem


def plot_descriptors_map(
    xaxis: np.ndarray,
    yaxis: np.ndarray,
    values: np.ndarray,
    title_map: str = None,
    filename: str = None,
    coordenate_axes: Tuple[float, float] = None,
    axis: Tuple[int, int] = (0, 1),
    log_scale: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Función que dada una malla de puntos definida por xaxis e yaxis pinta
    un mapa de contorno con los valores de la malla values.

    Parameters
    ----------
    xaxis: np.ndarray
        Malla de puntos que define el eje x de los puntos. Array
        de dimensión (nx_points, ny_points).
    yaxis: np.ndarray
        Malla de puntos que define el eje y de los puntos. Array
        de dimensión (nx_points, ny_points).
    values: np.ndarray
        Malla de puntos que define el color de los puntos. Array
        de dimensión (nx_points, ny_points).
    title_map: str, default None
        Título de la gráfica.
    filename: str = None
        Nombre del archivo con el que se quiere guardar.
        En caso de que se pase None no se guarda la gráfica.
    coordenate_axes: Tuple[float, float] = None
        Tupla con los valores de los ejes que se quieren dibujar.
        En caso de None no se dibuja.

    Returns
    --------
    fig: plt.Figure
        Figura con el mapa de contorno.
    ax: plt.Axes
        Objeto Axes de la gráfica.
    """
    n_var = len(values.shape)
    x_values = xaxis
    y_values = yaxis
    z_values = values
    if n_var > 2:
        index_values = np.arange(n_var)
        index_values[0] = 1
        index_values[1] = 0
        mask_axis_plot = [0] * n_var
        mask_axis_plot[index_values[axis[0]]] = np.s_[0:-1]
        mask_axis_plot[index_values[axis[1]]] = np.s_[0:-1]
        x_values = xaxis[tuple(mask_axis_plot)]
        y_values = yaxis[tuple(mask_axis_plot)]
        z_values = values[tuple(mask_axis_plot)]
    fig, ax = plt.subplots(1, 1)
    title = ""
    if isinstance(filename, str):
        title = filename.split("/")[-1].split(".")[0]
    title = title_map or title
    plt.title(title)
    if log_scale:
        z_values = np.log10(z_values)
    cp = ax.contourf(x_values, y_values, z_values)
    norm = Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    # a previous version of this used
    # norm= matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    # which does not work any more
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cp.cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ticks=cp.levels)

    z_label = ""
    if log_scale:
        z_label = "Log_10"
    cbar.set_label(z_label)
    ax.set_xlabel(f"x{axis[0]}-values")
    ax.set_ylabel(f"x{axis[1]}-values")
    if coordenate_axes:
        plt.vlines(coordenate_axes[0], y_values.min(), y_values.max())
        plt.hlines(coordenate_axes[1], x_values.min(), x_values.max())
    plt.savefig(filename)
    return fig, ax


def plot_extremals_solutions(
    ax: plt.Axes,
    extremals_functionals: Callable,
    args_func: Sequence[Any],
    x_min: np.ndarray,
    x_max: np.ndarray,
    t1: Union[int, float],
    diff_system: DiffSystem,
    solver_method: str,
) -> plt.Axes:
    x_pc, x_array, y_dy, y_pc = extremals_functionals(x_min, x_max, args_func)
    ax.scatter(x_pc, y_pc, color="r")
    ax.plot(x_array, y_dy, "r-")
    pert_y = -1e-3
    pert_x = 1e-2
    for i in range(x_pc.size):
        ds = diff_system
        _, _ = ds.solve_function(
            solver_method,
            [0, 2 * t1],
            np.array([x_pc[i] + pert_x, y_pc[i] + pert_y]),
            args_func,
        )
        t_ar = np.linspace(0, 2 * t1, 200)
        sol = ds.variable(t_ar).T
        ax.plot(sol[:, 0], sol[:, 1])
    ax.set_xlim(left=x_min[0], right=x_max[0])
    ax.set_ylim(bottom=x_min[1], top=x_max[1])
    return ax


def plot_poincare_sections(
    values: Union[np.ndarray, List[np.ndarray]],
    title_map: str = None,
    filename: str = None,
    axis: Tuple[int, int] = None,
) -> plt.Figure:
    if isinstance(values, np.ndarray):
        values = [values]
    if axis is None:
        axis = (0, 1)

    title = ""
    if isinstance(filename, str):
        title = filename.split("/")[-1].split(".")[0]
    title = title_map or title
    fig = plt.figure()
    plt.title(title)
    for i, var in enumerate(values):
        x = var[:, axis[0]]
        y = var[:, axis[1]]
        plt.scatter(x, y, s=3)
    plt.xlabel(f"x{axis[0]}-values")
    plt.ylabel(f"x{axis[1]}-values")
    plt.savefig(filename)
    return fig
