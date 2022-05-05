import numpy as np
from typing import Literal

import uq4pk_fit.cgn as cgn
from uq4pk_fit.special_operators import DiscreteLaplacian
from .minimize_surface import minimize_surface


def first_order_blanket(lb: np.ndarray, ub: np.ndarray, mode: Literal["fast", "exact"] = "fast"):
    """
    Computes the second-order blanket between two d-dimensional signals. That is, the optimization problem

    .. math::
        min_f \\qquad \\sum_{ij} \\sqrt{1 + ||\\nabla G f_{ij}||_2^2}

        s. t. \\qquad l \\leq f \\leq u.

    is solved.

    :param lb: The lower bound.
    :param ub: The upper bound.
    :param mode: The mode for solution. If "fast", then the simpler optimization problem

        .. math::
            min_f \\qquad \\sum_{ij} ||\\nabla \\delta f_{ij}||_2^2
            s. t. \\qquad l \\leq f \\leq u.

        is solved.

    :return: The solution, as d-dimensional array of the same shape as 'lb' and 'ub'.
    """
    assert lb.shape == ub.shape

    # Initialize the discrete Laplacian.
    delta = DiscreteLaplacian(shape=lb.shape).mat

    def fun(f):
        """
        :math:`F(f) = \\Delta G f`
        """
        return delta @ f

    def jac(f):
        """
        :math:`F'(f) = \\nabla G`.
        """
        return delta

    lbvec = lb.flatten()
    ubvec = ub.flatten()
    x0 = 0.5 * (lbvec + ubvec)

    # Solve problem with CGN.
    n = lbvec.size
    x = cgn.Parameter(dim=n, name="x")
    # Add very small regularization term
    scale = np.sum(np.square(fun(ubvec)))
    x.beta = 1e-5 * scale
    x.lb = lbvec
    x.ub = ubvec
    problem = cgn.Problem(parameters=[x], fun=fun, jac=jac)
    solver = cgn.CGN()
    solution = solver.solve(problem=problem, starting_values=[x0])
    x_min = solution.minimizer("x")

    # Bring minimizer into the correct format.
    x_arr = np.reshape(x_min, lb.shape)

    # Return the solution as two-dimensional numpy array.
    return x_arr