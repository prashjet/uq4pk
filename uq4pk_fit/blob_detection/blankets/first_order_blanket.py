import numpy as np

import uq4pk_fit.cgn as cgn
from uq4pk_fit.special_operators import DiscreteLaplacian


def first_order_blanket(lb: np.ndarray, ub: np.ndarray):
    """
    Computes the second-order blanket between two d-dimensional signals. That is, the optimization problem

    .. math::
        min_f || \\Delta f ||_2^2

        s. t. \\qquad l \\leq f \\leq u.

    is solved.

    :param lb: The lower bound.
    :param ub: The upper bound.

    :return: The solution, as d-dimensional array of the same shape as 'lb' and 'ub'.
    """
    assert lb.shape == ub.shape
    assert np.all(lb <= ub)
    # If ub > lb, then we can safely return the zero blanket.
    if ub.min() > lb.max():
        blanket = np.ones(lb.shape) * lb.max()
        return blanket

    # Initialize the discrete Laplacian.
    delta = DiscreteLaplacian(shape=lb.shape).mat

    def fun(f):
        """
        :math:`F(f) = \\Delta f`
        """
        return delta @ f

    def jac(f):
        """
        :math:`F'(f) = \\Delta`.
        """
        return delta

    lbvec = lb.flatten()
    ubvec = ub.flatten()
    x0 = 0.5 * (lbvec + ubvec)

    # Solve problem with CGN.
    n = lbvec.size
    x = cgn.Parameter(dim=n, name="x")
    x.lb = lbvec
    x.ub = ubvec
    problem = cgn.Problem(parameters=[x], fun=fun, jac=jac)
    solver = cgn.CGN()
    solver.options.ctol = 1e-8
    solver.options.set_verbosity(lvl=0)
    solution = solver.solve(problem=problem, starting_values=[x0])
    x_min = solution.minimizer("x")

    # Bring minimizer into the correct format.
    blanket = np.reshape(x_min, lb.shape)

    # Return the solution as two-dimensional numpy array.
    return blanket