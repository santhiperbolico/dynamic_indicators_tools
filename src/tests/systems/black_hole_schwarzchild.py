from typing import Sequence, Union

import numpy as np

from dynamic_indicators_tools.differentials_systems.diff_system import DiffVariable


def function_system(
    v: DiffVariable, xc: Union[float, np.ndarray], k: float, w: float
) -> np.ndarray:
    x = v.values[:, [1]]
    y = v.values[:, [2]]
    px = v.values[:, [3]]
    py = v.values[:, [4]]
    phi = np.sqrt(2 * k * x / (1 + py**2 + 2 * k * x * px**2))
    fval0 = 2 * k * x * px * phi
    fval1 = py * phi
    fval2 = -(4 * k * x * px**2 + py**2 + 1) / (2 * x) * phi - w**2 * (x - xc)
    fval3 = -(w**2) * y
    return np.concatenate((fval0, fval1, fval2, fval3), axis=1)


def get_partials_functions_phi(v: DiffVariable, k: float):
    x = v.values[:, [1]]
    y = v.values[:, [2]]
    px = v.values[:, [3]]
    py = v.values[:, [4]]
    phi = np.sqrt(2 * k * x / (1 + py**2 + 2 * k * px**2 * x))
    partial0 = (1 + py**2) / (4 * k * x**2) * phi**3
    partial1 = np.zeros(y.shape)
    partial2 = -px * phi**3
    partial3 = -py / (2 * k * x) * phi**3
    return partial0, partial1, partial2, partial3


def get_partials_functions_systems(v: DiffVariable, k: float, w: float) -> np.ndarray:
    x = v.values[:, [1]]
    y = v.values[:, [2]]
    px = v.values[:, [3]]
    py = v.values[:, [4]]

    phi = np.sqrt(2 * k * x / (1 + py**2 + 2 * k * px**2 * x))
    phi_0, phi_1, phi_2, phi_3 = get_partials_functions_phi(v, k)

    partial0_0 = 2 * k * px * (phi + x * phi_0)
    partial0_1 = phi_1
    partial0_2 = 2 * k * x * (phi + px * phi_2)
    partial0_3 = 2 * k * x * px * phi_3

    partial1_0 = py * phi_0
    partial1_1 = phi_1
    partial1_2 = py * phi_2
    partial1_3 = phi + py * phi_3

    partial2_0 = (
        (py**2 + 1) / (2 * x**2) * phi
        - (4 * k * px**2 + py**2 + 1) / (2 * x) * phi_0
        - w**2
    )
    partial2_1 = phi_1
    partial2_2 = -4 * k * px * phi - (4 * k * px**2 + py**2 + 1) / (2 * x) * phi_2
    partial2_3 = -py / x * phi - (4 * k * px**2 + py**2 + 1) / (2 * x) * phi_3

    partial3_0 = np.zeros(x.shape)
    partial3_1 = -np.ones(y.shape) * w**2
    partial3_2 = np.zeros(px.shape)
    partial3_3 = np.zeros(py.shape)

    jacobian = np.array(
        [
            [partial0_0, partial1_0, partial2_0, partial3_0],
            [partial0_1, partial1_1, partial2_1, partial3_1],
            [partial0_2, partial1_2, partial2_2, partial3_2],
            [partial0_3, partial1_3, partial2_3, partial3_3],
        ]
    ).reshape((4, 4, v.values.shape[0]))
    return jacobian


def function_system_equation_variational(
    v: DiffVariable, xc: Union[float, np.ndarray], k: float, w: float
) -> np.ndarray:
    jacobian = v.values[:, 5:].reshape((4, 4, v.values.shape[0]))
    jacobian_f = get_partials_functions_systems(v, k, w)
    fval = np.zeros(v.values[:, 1:].shape)
    fval[:, :4] = function_system(v, xc, k, w)
    for i in range(fval.shape[0]):
        fval[i, 4:] = np.dot(jacobian_f[:, :, i], jacobian[:, :, i]).reshape(16)
    return fval


def extremals_functionals(
    x_min: np.ndarray, x_max: np.ndarray, xc: np.ndarray, k: float, w: float, n_points: int = 500
) -> Sequence[np.ndarray]:
    # Cálculo para las curvas que minimizan el funcional del descriptor lagrangiano.
    x_array = np.linspace(x_min[0], x_max[0], n_points)
    y_array = np.linspace(x_min[1], x_max[1], n_points)
    v_x = w * w / 2 * ((x_array - xc) ** 2 + y_array**2)
    return [x_array, v_x]


def hamiltonian_py(
    h0: float, xc: Union[np.ndarray, float], k: float, w: float, epsilon: float = 0
):
    """
    Función que devuelve el valor de py para un valor h0 dado del hamiltoniano.
    """

    def projection(x: np.ndarray, y: np.ndarray, px: np.ndarray) -> np.ndarray:
        v_pot = (w**2) * ((x - xc) ** 2 + y**2) / 2
        coef_hp = (h0 - v_pot) ** 2 / (2 * k * x)
        r = 2 * k * x * px**2
        # py^4 + py^2(2*(r-1) - coef_hp) - coef_hp * (r+1) + (r-1) ^2 = 0
        b2 = 2 * (r - 1) - coef_hp
        c2 = -coef_hp * (r + 1) + (r - 1) ** 2
        py2 = (-b2 + np.sqrt(b2**2 - 4 * c2)) / 2
        py = np.sqrt(py2)
        return py

    return projection


def hamiltonian(xc: Union[np.ndarray, float], k: float, w: float):
    """
    Función que devuelve el valor de py para un valor h0 dado del hamiltoniano.
    """

    def projection(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        phi = np.sqrt(2 * k * x / (1 + py**2 + 2 * k * x * px**2))
        coef = py**2 + 2 * k * x * px**2 - 1
        v_pot = (w**2) * ((x - xc) ** 2 + y**2) / 2
        return phi * coef + v_pot

    return projection
