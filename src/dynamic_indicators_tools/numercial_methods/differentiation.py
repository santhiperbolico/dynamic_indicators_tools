from typing import Callable, NamedTuple, Sequence, Union

import numpy as np

# DERIVADAS NUMÉRICAS IMPLEMENTADAS PUNTO A PUNTO


class ParamsDiffMethods(NamedTuple):
    """
    Parámetros para las funciones asociadas a los métodos
    de diferenciación numérica punto a punto.

    Parameters
    -----------
    function: Callable[[np.ndarray], np.ndarray]
        Función para derivar
    x_point: np.ndarray
        Array con los puntos donde se quiere calcular la derivada.
        La dimension de este array es (n_obs, n_variable).
    h_step: np.ndarray
        Array (1, n_variable) con el paso utilizado.
    partial_i: int
        Componente de la derivada parcial.
    """

    function: Callable[[np.ndarray], np.ndarray]
    x_point: np.ndarray
    h_step: np.ndarray
    partial_i: int


def diff_num_1degree(params: ParamsDiffMethods) -> np.ndarray:
    """
    Función que implementa la deriavada numérica de diferencias
    hacia delante.

    Parameters
    ----------
    params: ParamsDiffMethods

    Returns
    -------
    d_function: np.ndarray
        Array con los valores de la derivada
    """
    function = params.function
    x_point = params.x_point
    partial_i = params.partial_i

    h_step_i = np.zeros((1, x_point.shape[1]))
    h_step_i[0, partial_i] = params.h_step[partial_i]
    x_point_h = x_point + h_step_i
    function_val = function(x_point)
    function_val_h = function(x_point_h)
    d_function = np.zeros(function_val.shape)
    for i in range(function_val.shape[0]):
        d_function[i] = (function_val_h[i] - function_val[i]) / h_step_i[0, partial_i]
    return d_function


def diff_num_2degree(params: ParamsDiffMethods) -> np.ndarray:
    """
    Función que implementa la deriavada numérica de diferencias
    centrales.

    Parameters
    ----------
    params: ParamsDiffMethods

    Returns
    -------
    d_function: np.ndarray
        Array con los valores de la derivada
    """
    function = params.function
    x_point = params.x_point
    partial_i = params.partial_i
    h_step_i = np.zeros((1, x_point.shape[1]))
    h_step_i[0, partial_i] = params.h_step[partial_i]
    x_point_h1 = x_point + h_step_i
    x_point_h0 = x_point - h_step_i
    function_val_h1 = function(x_point_h1)
    function_val_h0 = function(x_point_h0)
    d_function = np.zeros(function_val_h1.shape)
    for i in range(x_point.shape[0]):
        d_function[i] = (function_val_h1[i] - function_val_h0[i]) / h_step_i[0, partial_i] / 2
    return d_function


def diff_num_function(
    function: Callable[[np.ndarray], np.ndarray],
    x_point: np.ndarray,
    h_step: np.ndarray,
    partial_i: int,
    n_degree: int = 2,
) -> np.ndarray:
    """
     Función que implementa la deriavada numérica indicada por n_degree.

     Parameters
     ----------
     function: Callable[[np.ndarray], np.ndarray]
         Función para derivar
     x_point: np.ndarray
         Array con los puntos donde se quiere calcular la derivada.
         La dimension de este array es (n_obs, n_variable).
     h_step: np.ndarray
         Array (1, n_variable) con el paso utilizado.
     partial_i: int
         Componente de la derivada parcial.
     n_degree: int
         Grado del error de la derivada numérica

     Returns
     -------
     Returns
    -------
    d_function: np.ndarray
        Array con los valores de la derivada

    """
    x = x_point
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)

    dic_diff_functions = {1: diff_num_1degree, 2: diff_num_2degree}
    try:
        diff_function = dic_diff_functions[n_degree]
    except KeyError:
        raise ValueError(
            f"No está implementado la diferenciación de grado {n_degree}."
            f" Prueba con los grados {list(dic_diff_functions.keys())}"
        )
    params = ParamsDiffMethods(function, x, h_step, partial_i)
    return diff_function(params)


def jacobian_matrix(
    function: Callable[[np.ndarray], np.ndarray],
    x_point: np.ndarray,
    h_step: Union[int, float, np.ndarray],
    n_degree: int = 2,
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Función que devuelva una lista con las matrices jacobianas
    de los puntos x_point. Si solo hay un punto devuelve directamente
    la matriz jacobiana.

    Parameters
    ----------
    function: Callable[[np.ndarray], np.ndarray]
         Función para calcular la jacobiana
    x_point: np.ndarray
     Array con los puntos donde se quiere calcular la jacobiana.
     La dimension de este array es (n_obs, n_variable).
    h_step: Union[int, float, np.ndarray],
     Array (1, n_variable) con el paso utilizado.
    n_degree: int
     Grado del error de la derivada numérica

    Returns
    -------
    jacobians: Union[np.ndarray, Sequence[np.ndarray]]
        lista con las matrices jacobianas de los puntos x_point.
        Si solo hay un punto devuelve un array con la matriz jacobiana.
    """
    x = x_point
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    x_dim = x.shape[1]

    if isinstance(h_step, (float, int)):
        h_step = np.ones(x_dim) * h_step

    jacobians = []
    fval0 = function(x[0, :])
    for k in range(x.shape[0]):
        jac = np.zeros((x_dim, fval0.shape[1]))
        for partial_i in range(x_dim):
            jac[partial_i, :] = diff_num_function(function, x[[k], :], h_step, partial_i, n_degree)
        jacobians.append(jac)
    if len(jacobians) == 1:
        return jacobians[0]
    return jacobians


# DERIVADAS NUMÉRICAS IMPLEMENTADAS EN UNA MALLA


def diff_partials_grid(
    function_values: np.ndarray,
    n_var: int,
    h_steps: np.ndarray,
    edge_remove: bool,
) -> Sequence[np.ndarray]:
    """
    Función que calcula las derivadas parcia
    Parameters
    ----------
    function_values: np.ndarray
        Malla con los valores de la función
    n_var: int
        Número de variables o dimensión de la variable.
    h_steps: np.ndarray
        Array con los pasos de la malla por dimensión.
    edge_remove: bool
        Inidica si se quiere borrar los brodes de la malla.

    Returns
    -------

    """
    gradient_axis = np.arange(n_var)
    gradient_axis[0:2] = gradient_axis[1::-1]
    mask = (np.s_[:],) * n_var
    if edge_remove:
        mask = (np.s_[1:-1],) * n_var
    return [
        np.gradient(function_values, h_steps[i], axis=gradient_axis[i])[mask] for i in range(n_var)
    ]
