import numpy as np

from ..discretization import AdaptiveDiscretization
from ..evaluation import AffineEvaluationMap
from ..filter import FilterFunction
from .filter_functional import FilterFunctional


def filter_function_to_evaluation_map(filter_function: FilterFunction, discretization: AdaptiveDiscretization,
                                      x_map: np.ndarray, weights: np.ndarray = None) -> AffineEvaluationMap:
    """
    Takes a FilterFunction object and from it creates an AffineEvaluationMap object.

    :param weights: If provided, then the image is rescaled with this weight vector before it is evaluated.
    """
    # For each filter in the filter function, compute the associated affine evaluation functional.
    aef_list = []
    for filter, discr in zip(filter_function.get_filter_list(), discretization.discretizations):
        aef = FilterFunctional(filter, discr, x_map, weights)
        aef_list.append(aef)
    # Create the affine evaluation map from the list of evaluation functionals.
    aff_eval_map = AffineEvaluationMap(aef_list)
    # Sanity check, then return.
    _check_affine_evaluation_map(aff_eval_map, filter_function)
    return aff_eval_map


def _check_affine_evaluation_map(aemap: AffineEvaluationMap, filter_function: FilterFunction):
    """
    Checks that an affine evaluation map truly corresponds to a filter function.
    """
    # Check that the dimensions are the same
    assert aemap.dim == filter_function.dim


