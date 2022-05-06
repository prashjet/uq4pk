
import numpy as np

from ..evaluation import AffineEvaluationMap
from ..filter import FilterFunction
from .filter_functional import FilterFunctional


def filter_to_evaluation_map(filter_function: FilterFunction, x_map: np.ndarray)\
        -> AffineEvaluationMap:
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
    assert aff_eval_map.dim == filter_function.dim
    return aff_eval_map
