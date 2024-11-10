import numpy as np
import pytest

from dynamic_indicators_tools.numercial_methods.root_methods import newton_null_point


def function1d():
    def inner(x: np.ndarray) -> np.ndarray:
        return x - np.cos(x)

    return inner


def jacfunction1d():
    def inner(x: np.ndarray) -> np.ndarray:
        return 1 + np.sin(x)

    return inner


def function2d():
    def inner(x: np.ndarray) -> np.ndarray:
        fx0 = x[0] - np.exp(x[0]) + 1
        fx1 = x[1] - np.cos(x[1])
        return np.array([fx0, fx1])

    return inner


def jacfunction2d():
    def inner(x: np.ndarray) -> np.ndarray:
        jac = np.array([[1 - np.exp(x[0]), 0], [0, 1 + np.cos(x[1])]])
        return jac

    return inner


@pytest.mark.parametrize(
    "function, jac, expected",
    [
        (function1d(), jacfunction1d(), np.array([0.7390851332151607])),
        (function2d(), jacfunction2d(), np.array([0, 0.7390851332151607])),
    ],
)
def test_newton_null_point(function, jac, expected):
    tol = 1e-7
    x0 = np.ones(expected.shape)
    result = newton_null_point(function, jac, x0, tol=tol)
    assert result == pytest.approx(expected, abs=tol)
