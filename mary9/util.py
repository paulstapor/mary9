import numpy as np
import copy
from typing import Sequence, List


def rescale_to_bounds(
        parameters: np.ndarray,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float]
) -> np.ndarray:
    scaled_parameters = copy.copy(parameters)
    for i_par, par in enumerate(parameters):
        scaled_parameters[i_par] = \
            (upper_bounds[i_par] - lower_bounds[i_par]) * par \
            + lower_bounds[i_par]

    return scaled_parameters


def unscale_from_bounds(parameters: List[np.ndarray],
                        lower_bounds: Sequence[float],
                        upper_bounds: Sequence[float]) -> List[np.ndarray]:
    unscaled_parameters = copy.copy(parameters)
    for ip, par in enumerate(parameters):
        unscaled_parameters[ip] = (par - lower_bounds[ip]) / \
                                  (upper_bounds[ip] - lower_bounds[ip])
    return unscaled_parameters


def rescale_to_bounds_ensemble(
        parameters: List[np.ndarray],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float]
) -> List[np.ndarray]:
    rescaled_parameters = []
    n_parameters = len(lower_bounds)
    for par in parameters:
        rescaled_parameters.append(
            np.array([lower_bounds[i_par] +
                      (upper_bounds[i_par] - lower_bounds[i_par]) * par[i_par]
                      for i_par in range(n_parameters)
                      ]).flatten()
        )

    return rescaled_parameters


def unscale_from_bounds_ensemble(
        parameters: List[np.ndarray],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float]
) -> List[np.ndarray]:
    unscaled_parameters = []
    n_parameters = len(lower_bounds)
    for par in parameters:
        unscaled_parameters.append(
            np.array([(par[ip] - lower_bounds[ip]) /
                      (upper_bounds[ip] - lower_bounds[ip])
                      for ip in range(n_parameters)]).flatten()
        )

    return unscaled_parameters
