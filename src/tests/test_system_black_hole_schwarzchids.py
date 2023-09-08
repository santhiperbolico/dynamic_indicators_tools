import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable
from tests.systems.black_hole_schwarzchild import (
    function_system,
    get_partials_functions_phi,
    hamiltonian,
    hamiltonian_py,
)


@pytest.mark.parametrize(
    "values, xc, k, w, expected",
    [
        (np.array([[1, 0, 0, 0]]), 1.0, 0.5, 10, np.array([[0, 0, -0.5, 0]])),
        (
            np.array([[1, 1, 1, 0]]),
            1.0,
            0.5,
            10,
            np.array([[np.sqrt(1 / 2), 0, -3 / np.sqrt(8), -100]]),
        ),
        (
            np.array([[0.5, 3, np.sqrt(0.5), 1]]),
            0,
            2,
            5,
            np.array(
                [
                    [
                        2 / np.sqrt(3),
                        np.sqrt(2 / 3),
                        -4 * np.sqrt(2 / 3) - 25 / 2,
                        -75,
                    ]
                ]
            ),
        ),
    ],
)
def test_function_system(values: np.ndarray, xc: float, k: float, w: float, expected: np.ndarray):
    """
    Test que comprueba las ecuaciones de movimiento de un oscilador orbitando alrededor de un
    sistema de agujero negro de Schwarzschild.
    Parameters
    ----------
    values: np.ndarray
        Valores de las coordenadas x, y, p_x, p_y del sistema.
    xc: float
        Valor de la coordenada x del centro del oscilador.
    k: float
        Valor de la gravedad superficial.
    w: float
        Valor de la frecuencia del oscilador lineal.
    expected: np.ndarray
        Valor esperado del sistema de las ecuaciones del movimiento.
    """
    v = DiffVariable("v")
    t = np.array([0])
    v.set_values(t, values)
    result = function_system(v, xc=xc, k=k, w=w)
    assert expected == pytest.approx(result, 1e-4)


@pytest.mark.parametrize(
    "values, k, expected",
    [
        (np.array([[1, 0, 0, 0]]), 0.5, np.array([1 / 2, 0, 0, 0])),
        (np.array([[1, 1, 1, 0]]), 0.5, np.array([np.sqrt(2) / 8, 0, -1 / np.sqrt(8), 0])),
        (
            np.array([[1, 1, 1, 1]]),
            0.5,
            np.array([np.sqrt(3) / 9, 0, -1 / np.sqrt(27), -1 / np.sqrt(27)]),
        ),
        (
            np.array([[0.5, 0, np.sqrt(0.5), 1]]),
            2,
            np.array([np.sqrt(3 / 2) * 4 / 9, 0, -2 / (np.sqrt(27)), -np.sqrt(2 / 27)]),
        ),
    ],
)
def test_get_partials_functions_phi(values, k, expected):
    """
    Test que verifica que las derivadas parciales de la función Phi del sistema
    de ecuaciones de movimiento de un oscilador orbitando alrededor de un
    sistema de agujero negro de Schwarzschild es correcto.

    Parameters
    ----------
    values: np.ndarray
        Valores de las coordenadas x, y, p_x, p_y del sistema.
    k: float
        Valor de la gravedad superficial.
    expected: np.ndarray
        Valores esperados de las derivadas parciales.
    """
    v = DiffVariable("v")
    t = np.array([0])
    v.set_values(t, values)
    result = np.array(get_partials_functions_phi(v, k)).reshape(4)
    assert result == pytest.approx(expected, 1e-4)


@pytest.mark.parametrize(
    "x, px, h0, xc, k, w",
    [
        (1, -6, 15, 1, 0.5, 10),
        (1.3, 5, 15, 1, 0.5, 10),
        (1.4, 2, 15, 1, 0.5, 10),
        (0.6, -2, 15, 1, 0.5, 10),
        (0.456, 0, 15, 1, 0.5, 10),
    ],
)
def test_projection_h0(x: float, px: float, h0: float, xc: float, k: float, w: float):
    """
    Test que comprueba la proyección hecha sobre el plano (x, px) de puntos del espacio de fase de
    un sistema diferencial de las ecuaciones de movimiento de un oscilador orbitando alrededor
    de un sistema de agujero negro de Schwarzschild. Esta proyección se hace considerando puntos
    con y=0 y p_y tal que el hamiltoniano es igual a h0.

    La función comprueba que el punto py generado devuelve un valor del hamiltoniano h0.

    Parameters
    ----------
    x: float
        Valor de la coordenada x
    px: float
        Valor de la coordenada px
    h0: float
        Valor del hamiltoniano.
    xc: float
        Valor de la coordenada x del centro del oscilador.
    k: float
        Valor de la gravedad superficial.
    w: float
        Valor de la frecuencia del oscilador lineal.
    """
    py = hamiltonian_py(h0, xc, k, w)(x, 0, px)
    h0_expected = hamiltonian(xc, k, w)(x, 0, px, py)
    assert h0 == pytest.approx(h0_expected, 1e-5)
