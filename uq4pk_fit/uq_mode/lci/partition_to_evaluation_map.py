
import numpy as np

from ..evaluation import AffineEvaluationMap
from ..discretization import Partition
from .lci_evaluation_functional import LCIEvaluationFunctional


def partition_to_evaluation_map(partition: Partition, x_map: np.ndarray) -> AffineEvaluationMap:
    """
    Given a discretization (and the MAP estimate), generates the EvaluationMap object needed for computing
    local credible intervals.
    """
    # Check input for consistency
    _check_input(partition, x_map)
    # For each discretization element, create the associated LCI-evaluation functional
    aefun_list = []
    for element in partition.get_element_list():
        aefun = LCIEvaluationFunctional(element, x_map)
        aefun_list.append(aefun)
    # Combine the evaluation functionals in an EvaluationMap object.
    affine_evaluation_map = AffineEvaluationMap(aef_list=aefun_list)
    # Return that EvaluationMap
    return affine_evaluation_map


def _check_input(partition, x_map):
    assert partition.dim == x_map.size
