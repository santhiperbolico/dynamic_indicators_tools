from typing import Any, Callable, Dict, NamedTuple, Sequence, Union

import numpy as np
from scipy.integrate import fixed_quad, newton_cotes, quad, quadrature

# Integradores de una variable t
DiferentialSystemFunction = Callable[[np.ndarray, np.ndarray, Sequence[Any]], np.ndarray]


class DoesntExistIntegrator(Exception):
    """
    Error que indica que no existe un integrador.
    """

    pass


class ParamsIntegrators(NamedTuple):
    func: Callable[[np.ndarray], np.ndarray]
    a: Union[float, int]
    b: Union[float, int]
    args: tuple
    params_integrator: Dict[str, Any]


class ResultIntegrators(NamedTuple):
    """
    Resultado esperado de un método integrador
    para una variable.

    Parameters
    -----------
    y: float
        Valor de la integral
    error: float
        Error numérico esperado de la integral
    """

    y: float
    error: float


def newton_cotes_integrator(
    func: DiferentialSystemFunction,
    a: Union[float, int],
    b: Union[float, int],
    args: tuple,
    rn: int = 10,
    equal: int = 0,
) -> ResultIntegrators:
    """
    Método que calcula los pesos de Newton-Cotes para calcular
    el valor de la integral de func en el intervalo a,b. Se utiliza
    el método newton-cotes de scipy.integrate.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.newton_cotes.html

    Parameters
    ----------
    func: DiferentialSystemFunction
        Función a integrar
    a: Union[float, int]
        Extremo inferior del intervalo de intergracoión.
    b: Union[float, int]
        Extremo superior del intervalo de intergracoión.
    args: tuple
        Argumentos de la función
    rn: int, default 10
        Orden del integrador, ver scipy.integrate.newton_cotes
    equal: int, default 0
        Póngase a 1 para forzar datos equiespaciados.
        Ver  scipy.integrate.newton_cotes

    Returns
    -------
    y: float
        Valor de la integral
    error: float
        Error numérico esperado de la integral
    """
    x = np.linspace(a, b, rn + 1)
    weights_nc, error_coef = newton_cotes(rn, equal)
    dx = (b - a) / rn
    y = dx * np.sum(weights_nc * func(x, *args))
    return ResultIntegrators(y, error_coef)


def generator_integrator(method: str) -> Callable[[ParamsIntegrators], ResultIntegrators]:
    """
    Función que genera métodos integradores según lo indicado por
    method.

    Parameters
    ----------
    method: str
        Nombre del método integrador

    Returns
    -------
    integrator: Callable[[ParamsIntegrators], ResultIntegrators]
        Método integrador.
    """
    dic_integrators = {
        "quad": quad,
        "newton_cotes": newton_cotes_integrator,
        "quadrature": quadrature,
        "fixed_quad": fixed_quad,
    }

    try:
        method_int = dic_integrators[method]
    except KeyError:
        raise DoesntExistIntegrator(
            f"El método {method} no está implementado como integrador."
            f"Prueba con alguno de: {list(dic_integrators.keys())}."
        )

    def integrator(params: ParamsIntegrators) -> ResultIntegrators:
        y, error = method_int(
            func=params.func, a=params.a, b=params.b, args=params.args, **params.params_integrator
        )
        return ResultIntegrators(y, error)

    return integrator
