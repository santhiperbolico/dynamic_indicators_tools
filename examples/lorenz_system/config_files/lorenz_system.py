import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable


def function_system(variable: DiffVariable, sigma: float, rho: float, beta: float) -> np.ndarray:
    """
    Función que construye el sistema diferencial de Lorenz
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

    Parameters
    ----------
    variable: DiffVariable
        Variables del sistema diferencial.
    sigma: float
        Coeficiente
    rho: float
        Coeficiente
    beta: float
        Coeficiente

    Returns
    -------
    dv: np.ndarray
        Valores de f(v, t)
    """
    x = variable.values[:, 1]
    y = variable.values[:, 2]
    z = variable.values[:, 3]
    return np.concatenate((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))


def function_system_variational_equations(
    variable: DiffVariable, sigma: float, rho: float, beta: float
) -> np.ndarray:
    """
    Función que construye el sistema de ecuaciones variacionales del sistema diferencial de Lorenz
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

    Parameters
    ----------
    variable: DiffVariable
        Variables del sistema diferencial.
    sigma: float
        Coeficiente
    rho: float
        Coeficiente
    beta: float
        Coeficiente

    Returns
    -------
    dv: np.ndarray
        Valores de f(v, t)
    """
    x = variable.values[:, 1]
    y = variable.values[:, 2]
    z = variable.values[:, 3]

    g0 = variable.values[:, 4]
    g1 = variable.values[:, 5]
    g2 = variable.values[:, 6]
    g3 = variable.values[:, 7]
    g4 = variable.values[:, 8]
    g5 = variable.values[:, 9]
    g6 = variable.values[:, 10]
    g7 = variable.values[:, 11]
    g8 = variable.values[:, 12]

    fval0 = (sigma * (y - x)).reshape(-1, 1)
    fval1 = (x * (rho - z) - y).reshape(-1, 1)
    fval2 = (x * y - beta * z).reshape(-1, 1)

    fval3 = (-sigma * g0 + (rho - z) * g3 + y * g6).reshape(-1, 1)
    fval4 = (-sigma * g1 + (rho - z) * g4 + y * g7).reshape(-1, 1)
    fval5 = (-sigma * g2 + (rho - z) * g5 + y * g8).reshape(-1, 1)

    fval6 = (sigma * g0 + (-1) * g3 + x * g6).reshape(-1, 1)
    fval7 = (sigma * g1 + (-1) * g4 + x * g7).reshape(-1, 1)
    fval8 = (sigma * g2 + (-1) * g5 + x * g8).reshape(-1, 1)

    fval9 = (-x * g3 - beta * g6).reshape(-1, 1)
    fval10 = (-x * g4 - beta * g7).reshape(-1, 1)
    fval11 = (-x * g5 - beta * g8).reshape(-1, 1)

    return np.concatenate(
        (fval0, fval1, fval2, fval3, fval4, fval5, fval6, fval7, fval8, fval9, fval10, fval11),
        axis=1,
    )


def poincare_map_function(x: np.ndarray):
    return x[0]
