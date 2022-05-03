
import numpy as np
from typing import Sequence

from ..ci_computer import compute_credible_intervals
from ..downsampling import Downsampling, NoDownsampling
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from ..discretization import AdaptiveDiscretization
from .fci_class import FCI
from .filter_function_to_evaluation_map import filter_function_to_evaluation_map


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, filter_functions: Sequence[FilterFunction],
        discretization: AdaptiveDiscretization, downsampling: Downsampling = None, options: dict = None) \
        -> FCI:
    """
    Computes filtered credible intervals using the Pereyra approximation.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param filter_functions: Sequence of filter functions (one for each scale).
    :param discretization: The underlying discretization.
    :param options: A dictionary with additional options.
            - "use_ray": If True, then the computation is parallelized with the Ray framework. Default is True.
            - "num_cpus": Number of CPUs used by Ray.
            - "solver": The optimization solver. "slsqp" for SLSQP, "ecos" for ECOS solver.
            - "detailed": If True, then the solver also outputs all local optimizers. Default is False.
    :returns: Object of type :py:class:`FCI`.
    """
    # Check the input.
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.dim,)
    assert filter_functions[0].dim == model.dim
    assert discretization.dim == model.dim

    if options is None:
        options = {}
    # Get the optional weights
    weights = options.setdefault("weights", None)
    # For each filter function, generate corresponding affine evaluation map.
    affine_evaluation_map_list = [filter_function_to_evaluation_map(ffunction, discretization, x_map, weights=weights)
                                  for ffunction in filter_functions]
    # If subsample is not None, kick out all pixels that are not in sample.
    sample = options.setdefault("sample", None)
    # If subsampling occurs, downsampling is deactivated by default.
    # Read off downsampling parameters.
    if sample is not None:
        [affine_evaluation_map.select(sample) for affine_evaluation_map in affine_evaluation_map_list]
        # If subsampling happens, downsampling is deactivated by default.
        downsampling = NoDownsampling(dim=model.dim)
    elif downsampling is not None:
        sample = downsampling.indices()
        [affine_evaluation_map.select(sample) for affine_evaluation_map in affine_evaluation_map_list]
    else:
        downsampling = NoDownsampling(dim=model.dim)
    lower_stack, upper_stack, time_avg = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map,
                                                                    aemap_list=affine_evaluation_map_list,
                                                                    options=options)
    # If downsampling took place, have to enlarge output (else, the NoDownsampling object will simply leave it
    # untouched).
    lower_stack_enlarged = downsampling.enlarge(lower_stack)
    upper_stack_enlarged = downsampling.enlarge(upper_stack)

    fci_obj = FCI(lower_stack=lower_stack_enlarged, upper_stack=upper_stack_enlarged, time_avg=time_avg)
    return fci_obj
