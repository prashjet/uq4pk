
import numpy as np
import scipy.optimize as sciopt

from uq4pk_fit.special_operators import DiscreteGradient, DiscreteLaplacian
import uq4pk_fit.cgn as cgn


def minimize_bumps_exactly(lb: np.ndarray, ub: np.ndarray, rtol=1e-14) -> np.ndarray:
    """
    Solves the optimization problem

    .. math::
        min_f \\qquad \\sum_{ij} \sqrt{1 + ||\\nabla G f_{ij}||_2^2}

        s. t. \\qquad l \\leq f \\leq u.

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :return: The minimizer of the optimization problem. An array of the same shape as ``lb`` and ``ub``.
    """
    _check_input(lb, ub)
    nabla = DiscreteGradient(lb.shape).mat
    delta = DiscreteLaplacian(lb.shape).mat
    nabla_delta_list = np.array_split(nabla @ delta, lb.ndim)
    ngtng_list = [nabla_g.T @ nabla_g for nabla_g in nabla_delta_list]

    def length(f):
        """
        The length (or hyper surface-area) of a graph f is given by
            l(f) = \\int_i \\sqrt(1 + \\nabla g f(x)^2)
        """
        grad_squared = sum([np.square(dg @ f) for dg in nabla_delta_list])
        return np.sum(np.sqrt(1 + grad_squared))

    def length_grad(f):
        """
        The gradient of the graph length is
            \\nabla l(f) = \\frac{\\nabla f}{\\sqrt{1 + \\nabla f^2}}.
        """
        numerator = sum([ngtng @ f for ngtng in ngtng_list])
        grad_squared = sum([np.square(nabla_g @ f) for nabla_g in nabla_delta_list])
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


def minimize_bumps(lb: np.ndarray, ub: np.ndarray, tol=1e-10, without_nabla = False) -> np.ndarray:
    """
    Solves the optimization problem:

    .. math::
        min_f \\qquad || \\nabla \Delta f(x)||_2^2 d x

        s. t.   \\qquad  l \\leq f \\leq u.

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :param tol: The tolerance for the optimization problem.
    :return: The minimizer ``f``. Has same shape as ``lb`` and ``ub``.
    """
    # Check input for consistency.
    _check_input(lb, ub)

    # Initialize differential operators.
    delta = DiscreteLaplacian(lb.shape).mat
    nabla = DiscreteGradient(lb.shape).mat
    if without_nabla:
        nabla_delta = delta
    else:
        nabla_delta = nabla @ delta

    def fun(f):
        """
        :math:`F(f) = \\nabla G f`
        """
        return nabla_delta @ f

    def jac(f):
        """
        :math:`F'(f) = \\nabla G`.
        """
        return nabla_delta

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
    solver.options.tol = tol
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

    :param x_im:
    :param lb:
    :param ub:
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
    return lb