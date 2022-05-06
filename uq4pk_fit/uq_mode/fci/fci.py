import numpy as np

from ..ci_computer import compute_credible_intervals
from ..downsampling import Downsampling, NoDownsampling
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from .fci_class import FCI
from .filter_to_evaluation_map import filter_to_evaluation_map


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, filter_function: FilterFunction, options: dict = None) \
        -> FCI:
    """
    Computes filtered credible intervals using the Pereyra approximation and adaptive discretization.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param filter_functions: Sequence of filter functions (one for each scale).
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
    assert filter_function.dim == model.dim

    if options is None:
        options = {}
    # For each filter function, generate corresponding affine evaluation map.
    affine_evaluation_map = filter_to_evaluation_map(filter_function, x_map)
    lower_stack, upper_stack, time_avg = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map,
                                                                    aemap_list=[affine_evaluation_map],
                                                                    options=options)

    fci_obj = FCI(lower_stack=lower_stack, upper_stack=upper_stack, time_avg=time_avg)
    return fci_obj
