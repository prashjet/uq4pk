
import cvxpy as cp
import numpy as np
import scipy.optimize as sciopt

from uq4pk_fit.special_operators import DiscreteGradient


def span_blanket(lb: np.ndarray, ub: np.ndarray, rtol: float = 1e-14) -> np.ndarray:
    """
    Spans an N-dimensional "blanket" between a lower and an upper bound.

    This is achieved by solving the optimization problem
        min_x \\sqrt(1 + ||\\nabla f(x)||^2) s.t. lb <= x <= ub.

    :param lb: Of shape (m, n).
    :param ub: Of shape (m, n).
    :param rtol: Relative tolerance for the optimization solver. The solver uses the absolute tolerance
        ftol = rtol * surf(ub), where surf(ub) is the N-dimensional surface of the ``ub`` array (e.g. for a
        1-dimensional array, this would be the length of the graph).
    :return: Of shape (m, n).
    """
    # Check input for consistency.
    _check_input(lb, ub)
    dg = DiscreteGradient(lb.shape).mat
    dg_list = np.array_split(dg, lb.ndim)
    dgtdg_list = [dg.T @ dg for dg in dg_list]
    def length(f):
        """
        The length (or hyper surface-area) of a graph f is given by
            l(f) = \\int_i \\sqrt(1 + \\nabla f(x)^2)
        """
        grad_squared = sum([np.square(dg @ f) for dg in dg_list])
        return np.sum(np.sqrt(1 + grad_squared))
    def length_grad(f):
        """
        The gradient of the graph length is
            \\nabla l(f) = \\frac{\\nabla f}{\\sqrt{1 + \\nabla f^2}}.
        """
        numerator = sum([dgtdg @ f for dgtdg in dgtdg_list])
        grad_squared = sum([np.square(dg @ f) for dg in dg_list])
        denominator = np.sqrt(1 + grad_squared)
        return numerator / denominator
    lbvec = lb.flatten()
    ubvec = ub.flatten()
    bnds = sciopt.Bounds(lb=lbvec, ub=ubvec)
    x0 = 0.5 * (lbvec + ubvec)
    # Set tolerance.
    surf_ub = length(ubvec)
    ftol = rtol * surf_ub
    x_min = sciopt.minimize(fun=length, jac=length_grad, bounds=bnds, x0=x0, method="SLSQP",
                            options={"ftol": ftol}).x
    x_arr = np.reshape(x_min, lb.shape)
    _sanity_check(x_arr, lb, ub)
    # Return the solution as two-dimensional numpy array.
    return x_arr


def _check_input(lb: np.ndarray, ub: np.ndarray):
    if lb.shape != ub.shape:
        raise Exception("'lb' and 'ub' must have same shape.")


def _sanity_check(x_im: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol=1e-6):
    """
    Performs basic sanity check, i.e. the solution must satisfy the constraints.

    :param x_im: Of shape (m, n).
    :param lb: Of shape (m, n).
    :param ub: Of shape (m, n).
    :param tol: The maximum error with which a constraint is allowed to be violated.
    :raises Exception: If x is infeasible.
    """
    lb_error = np.max((lb - x_im).clip(min=0.))
    ub_error = np.max((x_im - ub).clip(min=0.))
    if lb_error > tol:
        raise Warning("Unable to satisfy lower bound constraint."
                      f"Violation is {lb_error}, but tolerance is {tol}")
    if ub_error > tol:
        raise Warning("Unable to satisfy upper bound constraint."
                      f"Violation is {ub_error}, but tolerance is {tol}")



