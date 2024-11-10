from typing import Callable

import numpy as np


def inverse_product_generator(dimension: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Función que devuelve el producto de la función inversa de la derivada
    según la dimensión. Si la dimensión es 1 devuelve f(x_i) / f'(x_i), si la dimensión
    es 2 o mayor es Df(x_i)^{-1} * f(x_i)

    Parameters
    ----------
    dimension: int
        Dimensión de la función.

    Returns
    -------
    inverse_product: Callable[[np.ndarray, np.ndarray], np.ndarray]
        Función que devuelve el producto de f por la inversa de su diferencial.

    """

    if dimension > 1:

        def inverse_product(fxi: np.ndarray, dfxi: np.ndarray) -> np.ndarray:
            return np.dot(np.linalg.inv(dfxi), fxi)

        return inverse_product

    def inverse_product(fxi: np.ndarray, dfxi: np.ndarray) -> np.ndarray:
        return fxi / dfxi

    return inverse_product


def newton_null_point(
    vectorial_function: Callable[[np.ndarray], np.ndarray],
    jacobian_function: Callable[[np.ndarray], np.ndarray],
    x_0: np.ndarray,
    tol: float = 1e-3,
    max_iteration: int = 1000,
) -> np.ndarray:
    """
    Método de Newton para encontrar raices de una función vectorial.

    Parameters
    ----------
    vectorial_function: Callable[[np.ndarray], np.ndarray]
        Función vectorial de la que queremos encontrar sus raices.
    jacobian_function: Callable[[np.ndarray], np.ndarray]
        Función que devuelve la jacobianda de l afunción vectorial.
    x_0: np.ndarray
        Punto inicial del método.
    tol: float, default 1e-3
        Tolerancia del método. Cuando se alcanza dicha tolerancia el método para.
    max_iteration: int, default 1000
        Número de iteraciones máximas del método.

    Returns
    -------
    xi: np.ndarray
        Raiz encontrada.
    """
    inverse_product = inverse_product_generator(x_0.size)
    xi = x_0
    iter = 0
    atol = 1e3
    while iter < max_iteration and atol > tol:
        fxi = vectorial_function(xi)
        dfxi = jacobian_function(xi)
        xii = xi - inverse_product(fxi, dfxi)
        atol = np.linalg.norm(xii)
        xi = xii
        iter += 1
    return xi
