import numpy as np
from typing import Sequence

from ..ci_computer.stack_computer import StackComputer
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from ..discretization import TrivialAdaptiveDiscretization
from .fci_class import FCI
from .filter_function_to_evaluation_map import filter_function_to_evaluation_map


def fci_stack(alpha: float, model: LinearModel, x_map: np.ndarray, ffunction_list: Sequence[FilterFunction],
              options: dict = None) -> FCI:
    # Check the input.
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.dim,)
    for ffunction in ffunction_list:
        assert ffunction.dim == model.dim

    if options is None: options = {}
    # Get the optional weights
    weights = options.setdefault("weights", None)
    discretization = TrivialAdaptiveDiscretization(dim=model.dim)
    # Generate list of affine evaluation maps from the filter function.
    aemap_list = []
    for ffunction in ffunction_list:
        affine_evaluation_map = filter_function_to_evaluation_map(ffunction, discretization, x_map, weights=weights)
        aemap_list.append(affine_evaluation_map)
    # Compute the credible intervals (in phi-space)
    computer = StackComputer(alpha=alpha, model=model, x_map=x_map, aemap_list=aemap_list, scale=x_map.max())
    lower_stack, upper_stack = computer.compute_all()

    return FCI(lower_stack=lower_stack, upper_stack=upper_stack)