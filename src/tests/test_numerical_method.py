import numpy as np
import pytest

from dynamic_indicators_tools.numercial_methods.root_methods import (
    newton_fix_point,
    newton_null_point,
)


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
        jac = np.array([[1 - np.exp(x[0]), 0], [0, 1 + np.sin(x[1])]])
        return jac

    return inner


def functionfix():
    def inner(x: np.ndarray) -> np.ndarray:
        fx0 = np.exp(x[0]) - 1
        fx1 = np.cos(x[1])
        return np.array([fx0, fx1])

    return inner


def jacfunctionfix():
    def inner(x: np.ndarray) -> np.ndarray:
        jac = np.array([[np.exp(x[0]), 0], [0, -np.sin(x[1])]])
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
    assert result.success
    assert result.root == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize(
    "x_min, x_max, x_0", [(None, None, np.ones(2)), (np.array([-1, 0]), np.array([1, 2]), None)]
)
def test_newton_fix_point(x_min, x_max, x_0):
    expected = np.array([0, 0.7390851332151607])
    tol = 1e-7
    result = newton_fix_point(
        jacobian_function=jacfunctionfix(),
        vectorial_function=functionfix(),
        x_min=x_min,
        x_max=x_max,
        x_0=x_0,
        tol=tol,
    )
    assert result.success
    assert result.root == pytest.approx(expected, abs=tol)
