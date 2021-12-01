
import numpy as np
import scipy.optimize as sciopt

from uq4pk_fit.special_operators import DiscreteGradient
import uq4pk_fit.cgn as cgn


def cheap_blanket(lb: np.ndarray, ub: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """
    Solves the optimization problem
        min_f ||\\nabla f||_2^2 s.t. lb <= f <= ub

    :param lb:
    :param ub:
    :param rtol:
    :return:
    """
    # Check input for consistency.
    _check_input(lb, ub)
    dg = DiscreteGradient(lb.shape).mat
    def fun(f):
        """
            G(f) = \\nabla f
        """
        return dg @ f
    def jac(f):
        """
            \\nabla G(f) = \\nabla
        """
        return dg
    lbvec = lb.flatten()
    ubvec = ub.flatten()
    x0 = 0.5 * (lbvec + ubvec)
    # Set tolerance.
    ftol = rtol
    # Solve problem with CGN.
    n = lbvec.size
    x = cgn.Parameter(dim=n, name="x")
    x.lb = lbvec
    x.ub = ubvec
    problem = cgn.Problem(parameters=[x], fun=fun, jac=jac)
    solver = cgn.CGN()
    solver.options.set_verbosity(lvl=2)
    solution = solver.solve(problem=problem, starting_values=[x0])
    x_min = solution.minimizer("x")
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