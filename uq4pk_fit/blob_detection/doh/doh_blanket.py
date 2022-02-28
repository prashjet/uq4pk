
import numpy as np

import uq4pk_fit.cgn as cgn
from uq4pk_fit.special_operators import DiscreteGradient

from uq4pk_fit.blob_detection.extra import image_hessian


def hessian_operator(m: int, n: int):
    """
    Returns the operator of shape (m * dim, m * dim) that computes the Hessian of an image.

    :param m:
    :param n:
    :return: h11, h12, h22
    """
    dim = m * n
    basis = np.identity(dim)
    h11_list = []
    h22_list = []
    h12_list = []
    h21_list = []
    for column in basis.T:
        im = np.reshape(column, (m, n))
        h = image_hessian(im)
        h11_list.append(h[0][0].flatten())
        h12_list.append(h[0][1].flatten())
        h21_list.append(h[1][0].flatten())
        h22_list.append(h[1][1].flatten())
    h11 = np.column_stack(h11_list)
    h12 = np.column_stack(h12_list)
    h21 = np.column_stack(h21_list)
    h22 = np.column_stack(h22_list)
    return h11, h12, h21, h22


class DOH:

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.h11, self.h12, self.h21, self.h22 = hessian_operator(m, n)

    def fun(self, f: np.ndarray):
        dhf = (self.h11 @ f) * (self.h22 @ f) - (self.h21 @ f) * (self.h12 @ f)
        return dhf

    def jac(self, f: np.ndarray) -> np.ndarray:
        """
        Return the Jacobian of the operator det H f.

        :param f: The flattened image.
        :return: A matrix of shape (d, d), where d = m * dim.
        """
        j = (self.h11 @ f) * self.h22 + (self.h22 @ f) * self.h11 - (self.h21 @ f) * self.h12 - \
            (self.h12 @ f) * self.h21
        return j


def doh_blanket(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Solves the optimization problem:

    .. math::
        min_f \\qquad || \\nabla det H f(x)||_2^2 d x

        s. t.   \\qquad  l \\leq f \\leq u,

    where :math:`Hf` is the Hessian of f..

    :param lb: The lower bound.
    :param ub: The upper bound. Must have same shape as ``lb``.
    :param g: Optional matrix. Set equal to the identity if not provided.
    :param tol: The tolerance for the optimization problem.
    :return: The minimizer ``f``. Has same shape as ``lb`` and ``ub``.
    """
    assert lb.ndim == 2

    # Initialize differential operators.
    nabla = DiscreteGradient(lb.shape).mat

    # Initialize nonlinear DOH-operator.
    doh_op = DOH(m=lb.shape[0], n=lb.shape[1])

    scale = np.sqrt(np.sum(np.square(nabla @ doh_op.fun(lb.flatten()))))

    def fun(f):
        """
        :math:`F(f) = \\nabla G f`
        """
        return nabla @ doh_op.fun(f) / scale

    def jac(f):
        """
        :math:`F'(f) = \\nabla G`.
        """
        return nabla @ doh_op.jac(f) / scale

    lbvec = lb.flatten()
    ubvec = ub.flatten()
    x0 = 0.5 * (lbvec + ubvec)
    # Solve problem with CGN.
    n = lbvec.size
    x = cgn.Parameter(dim=n, name="x")
    # Add very small regularization term
    x.beta = 1e-5
    x.mean = np.zeros(n)
    x.lb = lbvec
    x.ub = ubvec
    problem = cgn.Problem(parameters=[x], fun=fun, jac=jac)
    solver = cgn.CGN()
    solver.options.tol = 1e-10
    solver.options.set_verbosity(lvl=2)
    solution = solver.solve(problem=problem, starting_values=[x0])
    x_min = solution.minimizer("x")
    x_arr = np.reshape(x_min, lb.shape)

    doh_min = 0.5 * np.linalg.norm(fun(x_min)) ** 2
    doh_start = 0.5 * np.linalg.norm(fun(x0)) ** 2
    print(f"doh_start = {doh_start}")
    print(f"doh_min = {doh_min}")


    # Return the solution as two-dimensional numpy array.
    return x_arr