"""
Contains function "lci".
"""

import numpy as np

from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from .fci_computer import FCIComputer


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction):
    """
    Computes filtered credible intervals.
    """
    _check_input(alpha, model, x_map, ffunction)
    # Compute credible intervals through optimization.
    fci_computer = FCIComputer(alpha, model, x_map, ffunction)
    credible_intervals = fci_computer.compute()
    return credible_intervals

def _check_input(alpha, model, x_map, ffunction):
    """
    Checks the input of "fci" for consistency.
    """
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.n, )
    assert ffunction.dim == model.n
