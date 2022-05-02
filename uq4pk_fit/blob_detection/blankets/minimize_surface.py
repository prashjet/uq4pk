
import numpy as np
import scipy.optimize as sciopt
from typing import Literal, Union

import uq4pk_fit.cgn as cgn
from uq4pk_fit.special_operators import DiscreteGradient


FTOL = 1e-15    # Tolerance for solver.
CTOL = 1e-6     # absolute tolerance for constraint violation


def minimize_surface(lb: np.ndarray, ub: np.ndarray, mode: Literal["exact", "fast"], g: np.ndarray = None) \
        -> np.ndarray:
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
            min_f \\qquad \\sum_{ij} ||\\nabla G f_{ij}||_2^2
            s. t. \\qquad l \\leq f \\leq u.

        is solved.
    :param g: An optional matrix with which the flattened image is multiplied.
    :return: The solution, as d-dimensional array of the same shape as 'lb' and 'ub'.
    """
    # CHECK INPUT
    assert lb.shape == ub.shape
    # Check that lb <= ub
    assert np.all(lb <= ub), "'lb <= ub' must hold exactly!"
    d = lb.size
    if g is not None:
        assert g.ndim == 2
        assert g.shape[1] == d

    # Compute blanket with the method of choice.
    if mode == "fast":
        blanket = _minimize_surface_fast(lb, ub, g)
    elif mode == "exact":
        blanket = _minimize_surface_exactly(lb, ub, g)
    else:
        raise KeyError("Unknown mode.")

    # Perform sanity check.
    _sanity_check(blanket, lb, ub)

    return blanket


def _minimize_surface_exactly(lb: np.ndarray, ub: np.ndarray, g: Union[np.ndarray, None]) -> np.ndarray:
    """
    Solves the optimization problem

    .. math::
        min_f \\qquad \\sum_{ij} \\sqrt{1 + ||\\nabla G f_{ij}||_2^2}

        s. t. \\qquad l \\leq f \\leq u.

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :param g: Additional operator.
    :return: The minimizer of the optimization problem. An array of the same shape as ``lb`` and ``ub``.
    """
    nabla = DiscreteGradient(lb.shape).mat
    if g is None:
        nabla_g = nabla
    else:
        nabla_g = nabla @ g
    nabla_g_list = np.array_split(nabla_g, lb.ndim)
    ngtng_list = [nabla_g.T @ nabla_g for nabla_g in nabla_g_list]

    def length(f):
        """
        The length (or hyper surface-area) of a graph f is given by
            l(f) = \\int_i \\sqrt(1 + \\nabla g f(x)^2)
        """
        grad_squared = sum([np.square(dg @ f) for dg in nabla_g_list])
        return np.sum(np.sqrt(1 + grad_squared))

    def length_grad(f):
        """
        The gradient of the graph length is
            \\nabla l(f) = \\frac{\\nabla f}{\\sqrt{1 + \\nabla f^2}}.
        """
        numerator = sum([ngtng @ f for ngtng in ngtng_list])
        grad_squared = sum([np.square(ng @ f) for ng in nabla_g_list])
        denominator = np.sqrt(1 + grad_squared)
        return numerator / denominator

    lbvec = lb.flatten()
    ubvec = ub.flatten()
    bnds = sciopt.Bounds(lb=lbvec, ub=ubvec)
    x0 = 0.5 * (lbvec + ubvec)

    # Solve with scipy's SLSQP
    x_min = sciopt.minimize(fun=length, jac=length_grad, bounds=bnds, x0=x0, method="SLSQP",
                            options={"ftol": FTOL}).x

    x_arr = np.reshape(x_min, lb.shape)

    return x_arr


def _minimize_surface_fast(lb: np.ndarray, ub: np.ndarray, g: Union[np.ndarray, None]) -> np.ndarray:
    """
    Solves the optimization problem:

    .. math::
        min_f \\qquad || \\nabla G f(x)||_2^2 d x

        s. t.   \\qquad  l \\leq f \\leq u.

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :return: The minimizer ``f``. Has same shape as ``lb`` and ``ub``.
    """
    nabla = DiscreteGradient(lb.shape).mat
    if g is not None:
        nabla_g = nabla @ g
    else:
        nabla_g = nabla

    def fun(f):
        """
        :math:`F(f) = \\nabla G f`
        """
        return nabla_g @ f

    def jac(f):
        """
        :math:`F'(f) = \\nabla G`.
        """
        return nabla_g

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


def _sanity_check(x_im: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """
    Performs basic sanity check, i.e. the solution must satisfy the constraints.

    :param x_im:
    :param lb:
    :param ub:
    :raises Exception: If x is infeasible.
    """
    lb_error = np.max((lb - x_im).clip(min=0.))
    ub_error = np.max((x_im - ub).clip(min=0.))
    if lb_error > CTOL:
        raise Warning("Unable to satisfy lower bound constraint."
                      f"Violation is {lb_error}, but tolerance is {CTOL}")
    if ub_error > CTOL:
        raise Warning("Unable to satisfy upper bound constraint."
                      f"Violation is {ub_error}, but tolerance is {CTOL}")
    return lb
