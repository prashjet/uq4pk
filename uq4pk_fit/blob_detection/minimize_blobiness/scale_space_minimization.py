
import cvxpy as cp
import numpy as np
from typing import List, Sequence, Union

from uq4pk_fit.special_operators import DiscreteGradient, DiscreteLaplacian
from uq4pk_fit.uq_mode.linear_model import CredibleRegion, LinearModel
from uq4pk_fit.blob_detection.minimize_blobiness.blob_operator import blob_operator


def scale_space_minimization(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray,
                             sigma_list: Sequence[Union[float, np.ndarray]]):
    """
    Computes the minimizer of the optimization problem

    .. math::
        min_f ||\\nabla \\Delta_{x}^h L||_2^2 + \\beta ||f||_2^2
        s.t. L(x, h) = G_h f, f \\idim C_{\\alpha},

    where :math:`G_h` is the isotropic Gaussian filter with scale parameter h.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The linear statistical model.
    :param x_map: The MAP estimate for ``model``.
    :return: f, blobs
        - f: Numpy array of shape (m, dim).
        - blobs: Numpy array of shape (k, 4), where each row corresponds to a blob of f and is of the form
            (s_x, s_y, i, j),
            where s_x, s_y are the stdev in x- and y-direction for the blob, and (i, j) its center.
    """
    # Check input for consistency.
    assert 0 < alpha < 1
    assert model.dim == m * n
    assert x_map.shape == (model.dim,)

    # Translate sigmas to scales
    scales = [0.5 * sigma ** 2 for sigma in sigma_list]

    # Create the discretized operator \\nabla \\Delta_{x}^h G_h.
    dg = DiscreteGradient((len(sigma_list), m, n)).mat
    blobby = blob_operator(scales, m, n)
    shape_operator = dg @ blobby

    # Create the credible region
    cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)

    x_min = solve_optimization_problem(a=shape_operator, c=cregion)

    # Reshape the optimizer into an image.
    f = np.reshape(x_min, (m, n))

    return f


BETA_REL = 1e-5


def solve_optimization_problem(a: np.ndarray, c: CredibleRegion)\
        -> np.ndarray:
    """
    Solves the problem:

    .. math::
        f_min = argmin_f ||A f||_2^2 s.t. f \\in C_\\alpha,
    where :math:`C_\\alpha` is a credible region.

    :param alpha:
    :param x_map:
    :param model:
    :param shape_operator:
    :returns: The minimizer :math:`f_min`.
    """
    # Set up the CVXPY-problem:
    n = c.dim
    x = cp.Variable(n)
    #x.value = c.x_map  # Setting initial guess.
    # add SCOP constraint (||T x - d||_2 <= sqrt(e) <=> ||T x - d||_2^2 <= e)
    sqrt_e = np.sqrt(c.e_tilde)
    constraints = [cp.SOC(sqrt_e, (c.t @ x - c.d_tilde))]
    # add equality constraint
    if c.equality_constrained:
        constraints += [c.a @ x == c.b]
    if c.bound_constrained:
        # Cvxpy cannot deal with infinite values. Hence, we have to translate the vector bound x >= lb
        # to the element-wise bound x[i] >= lb[i] for all i where lb[i] > - infinity
        lb = c.lb
        bounded_indices = np.where(lb > -np.inf)[0]
        if bounded_indices.size > 0:
            constraints += [x[bounded_indices] >= lb[bounded_indices]]

    # Perform QR decomposition on a to reduce dimension
    q, r0 = np.linalg.qr(a, mode="complete")
    r = r0[:n, :]

    scale = np.sum(np.square(r @ c.x_map))
    alpha = 1 / scale
    beta = 0.0001

    problem = cp.Problem(cp.Minimize(alpha * cp.sum_squares(r @ x) + beta * cp.sum_squares(x)), constraints)

    # Solve the problem
    problem.solve(warm_start=False, verbose=False, solver=cp.ECOS)

    # Return the minimizer.
    x_min = x.value

    print(f"Blobiness of map: {np.sum(np.square(r @ c.x_map))}")
    print(f"Blobiness of solution: {np.sum(np.square(r @ x_min))}")
    return x_min
