from typing import Callable, Dict, List, Sequence, Union

import numpy as np
from attr import attrib, attrs


@attrs
class Projection:
    """
    Clase que recoge la funcionalidad de la proyección. Estas operaciones reducen la dimensión xj
    del espacio en una unidad utilizando las dimensiones indicadas index_variables x0,x1,...xi
    mediante el hiperplano xj - f(x0,x1,...,xi)

    Parameters
    -----------
    index_variables: List[int]
        Lista de las dimensiones utilizadas que definen el hiperplano.
    function: Callable[[Sequence[np.ndarray]], np.ndarray]
        Función f del hiperplano.
    """

    index_variables = attrib(type=List[int], init=True)
    function = attrib(type=Callable[[Sequence[np.ndarray]], np.ndarray], init=True)


def project_grid_data(
    grid_points: Union[np.ndarray, List[np.ndarray]], projection_config: Dict[int, Projection]
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Función que dada una lista con una malla de puntos (donde cada elemento de la lista
    indica la dimensión) o un array, aplica las proyecciones definidas en projection_config.
    Parameters
    ----------
    grid_points: Union[np.ndarray, List[np.ndarray]]
        Lista con la malla de puntos o el array.
    projection_config: Dict[int, Projection]
        Diccionario que relaciona la dimensión de la proyección (key) con la proyección (values)

    Returns
    -------
    grid_points: Union[np.ndarray, List[np.ndarray]]
        Lista con la malla de puntos o el array con la proyección.
    """
    if projection_config is None:
        projection_config = {}
    for key, projection in projection_config.items():
        variables = [grid_points[index] for index in projection.index_variables]
        grid_points[key] = projection.function(*variables)
    return grid_points
