
import numpy as np
import cvxpy as cp

from uq4pk_fit.special_operators import DiscreteGradient, DiscreteLaplacian


def minimize_tv2(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Solves the optimization problem

    .. math::
        min_f ||\nabla \\Delta f||_1

        s. t. \\qquad l \\leq f \\leq u.

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :return: The minimizer of the optimization problem. An array of the same shape as ``lb`` and ``ub``.
    """
    nabla = DiscreteGradient(lb.shape).mat
    delta = DiscreteLaplacian(lb.shape).mat
    nabla_delta = nabla @ delta
    lbvec = lb.flatten()
    ubvec = ub.flatten()
    n = lbvec.size

    # Setup cvxpy problem
    x = cp.Variable(n)
    constraints = [x >= lbvec, x <= ubvec]
    problem = cp.Problem(cp.Minimize(cp.norm1(nabla_delta @ x)), constraints)
    x.value = lbvec

    # Solve with ECOS
    problem.solve(warm_start=True, solver=cp.ECOS)
    x_sol = x.value

    # Return the solution as two-dimensional numpy array.
    x_arr = np.reshape(x_sol, lb.shape)
    return x_arr