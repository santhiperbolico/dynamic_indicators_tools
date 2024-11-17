from typing import Callable, Optional

import numpy as np
from attr import attrs


@attrs(auto_attribs=True, repr=False)
class SummaryRoot:
    """
    Clase que recoje el resulta de la búsqueda de raices de una función.

    Attributes
    ----------
    method: str
        Nombre del método utlizado.
    success: bool
        Indica si el método ha convergido.
    root: np.ndarray
        Raiz encontrada por el método.
    iter: np.ndarray
        Iteración donde se ha encontrado la raiz.
    tol: np.ndarrat
        Tolerancia obtenida en el método.
    """

    method: str
    success: bool
    root: np.ndarray
    iter: int
    tol: float

    def __repr__(self):
        message_method = f"The method {self.method} hasn't converged.\n"

        if self.success:
            message_method = f"The method {self.method} has converged.\n"

        message = (
            f"{message_method}\t SUCCESS: {self.success}\n"
            f"\t ROOT: {self.root}\n"
            f"\t ITERATIONS: {self.iter}\n"
            f"\t TOLERANCE: {self.tol}\n"
        )

        return message


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
) -> SummaryRoot:
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
    result: SummaryRoot
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
        atol = np.linalg.norm(xii - xi)
        xi = xii
        iter += 1

    success = True
    if iter >= max_iteration:
        success = False

    result = SummaryRoot("Newton", success, xi, iter, atol)
    return result


def newton_fix_point(
    vectorial_function: Callable[[np.ndarray], np.ndarray],
    jacobian_function: Callable[[np.ndarray], np.ndarray],
    x_min: Optional[np.ndarray] = None,
    x_max: Optional[np.ndarray] = None,
    x_0: Optional[np.ndarray] = None,
    tol: float = 1e-3,
    max_iteration: int = 1000,
) -> SummaryRoot:
    """
    Función que dada una función se encarga de buscar puntos fijos dada una función
    y su diferencial.

    Parameters
    ----------
    vectorial_function: Callable[[np.ndarray], np.ndarray]
        Función vectorial de la que queremos encontrar sus raices.
    jacobian_function: Callable[[np.ndarray], np.ndarray]
        Función que devuelve la jacobianda de l afunción vectorial.
    x_min: Optional[np.ndarray] = None
        Valores mínimos para encontrar un punto de partída. Se deben indicar si no se añade
        un punto de partida.
    x_max: Optional[np.ndarray] = None
        Valores mínimos para encontrar un punto de partída. Se deben indicar si no se añade
        un punto de partida.
    x_0: Optional[np.ndarray] = None
        Punto inicial de búsqueda. Se debe indicar si se ha dejado como nulo x_min y x_max. Si
        no se indica x_0 se busca un valor aleatorio entre x_min y x_max.
    tol: float, default 1e-3
        Tolerancia del método. Cuando se alcanza dicha tolerancia el método para.
    max_iteration: int, default 1000
        Número de iteraciones máximas del método.

    Returns
    -------
    result: SummaryRoot
        Punto fijo encontrado.

    Raises
    ------
    ValueError: Salta si el método no ha encontrado ningún punto fijo.
    """

    def function(x: np.ndarray) -> np.ndarray:
        return x - vectorial_function(x)

    def jac_function(x: np.ndarray) -> np.ndarray:
        return np.eye(x.size) - jacobian_function(x)

    if not isinstance(x_0, (np.ndarray, int, float)):
        x_0 = np.random.uniform(x_min, x_max)
    result = newton_null_point(function, jac_function, x_0, tol=tol, max_iteration=max_iteration)
    return result
