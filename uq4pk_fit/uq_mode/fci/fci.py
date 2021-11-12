"""
Contains function "lci".
"""

import numpy as np

from ..ci_computer import compute_credible_intervals
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from .filter_function_to_evaluation_map import filter_function_to_evaluation_map


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction, options: dict = None):
    """
    Computes filtered credible intervals.
    """
    _check_input(alpha, model, x_map, ffunction)
    # Generate an affine evaluation map from the filter function
    affine_evaluation_map = filter_function_to_evaluation_map(ffunction, x_map)
    # Compute the credible intervals (in phi-space)
    xi = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map, aemap=affine_evaluation_map,
                                                    options=options)
    # Enlarge to credible intervals in x-space
    xi_low = xi[:, 0]
    xi_upp = xi[:, 1]
    x_low = ffunction.enlarge(xi_low)
    x_upp = ffunction.enlarge(xi_upp)
    credible_intervals = np.column_stack([x_low, x_upp])
    return credible_intervals


def _check_input(alpha, model, x_map, ffunction):
    """
    Checks the input of "fci" for consistency.
    """
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.n, )
    assert ffunction.dim == model.n
