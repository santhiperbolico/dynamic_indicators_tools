import numpy as np
import pytest

from dynamic_indicators_tools.differentials_systems.diff_system import DiffSystem, DiffVariable
from dynamic_indicators_tools.dynamic_indicators.poincare_maps import poincare_section_grid


@pytest.fixture
def lorentz_system():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    var = DiffVariable("x")

    def lorentz_function(variable: DiffVariable):
        x = variable.values[:, 1]
        y = variable.values[:, 2]
        z = variable.values[:, 3]
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    return DiffSystem(var, lorentz_function)


@pytest.fixture
def poincare_map():
    def function(states):
        return states[0]

    return function


def test_poincare_section_grid(lorentz_system, poincare_map):
    x0 = np.array([1.0, 1.0, 1.0])
    T = 40
    t_span = [0, T]
    n_points = T * 100
    result = poincare_section_grid(
        diff_system=lorentz_system,
        poincare_map=poincare_map,
        solver_method="solve_ivp",
        t_span=t_span,
        x0=x0,
        n_points=n_points,
    )

    expected = np.array(
        [
            0.51298646,
            14.17828829,
            14.95354847,
            15.74609354,
            16.51259702,
            18.0835094,
            20.36845722,
            22.63899511,
            24.18033758,
            24.94841309,
            27.4313825,
            32.97882687,
            34.50835275,
            35.27987797,
            36.09752353,
            36.8684677,
            39.12347803,
        ]
    )
    assert result == pytest.approx(expected, 1e-3)
