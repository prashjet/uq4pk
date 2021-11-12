import numpy as np

from ..evaluation import AffineEvaluationMap
from ..filter import FilterFunction
from .filter_functional import FilterFunctional


def filter_function_to_evaluation_map(filter_function: FilterFunction, x_map: np.ndarray) -> AffineEvaluationMap:
    """
    Takes a FilterFunction object and from it creates an AffineEvaluationMap object.
    """
    # For each filter in the filter function, compute the associated affine evaluation functional.
    aef_list = []
    for filter in filter_function.get_filter_list():
        aef = FilterFunctional(filter, x_map)
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
    # Check that the number of evaluation functionals is equal to the number of filters.
    assert aemap.size == filter_function.size


