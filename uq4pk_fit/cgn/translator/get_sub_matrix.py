
import numpy as np

from uq4pk_fit.cgn.problem.linear_constraint import LinearConstraint


def get_sub_matrix(constraint: LinearConstraint, i: int) -> np.ndarray:
    """
    Returns the sub-matrix of the constraint that corresponds to the i-th parameter in the constraint.
    """
    # Compute the position of the parameter in the concatenated parameter vector
    parameters = constraint.parameters
    if i not in range(len(parameters)):
        raise Exception(f"'i' must be between 0 and {len(parameters)}")
    pos_i = 0
    for j in range(i):
        pos_i += parameters[j].dim
    dim_i = parameters[i].dim
    # Get the sub-matrix
    a = constraint.a
    a_sub = a[:, pos_i:pos_i + dim_i]
    return a_sub