
import cvxpy as cp
import numpy as np
from typing import List, Sequence, Union
from skimage.filters import gaussian

from uq4pk_fit.special_operators import DiscreteGradient, DiscreteLaplacian
from uq4pk_fit.uq_mode.linear_model import CredibleRegion, LinearModel
from uq4pk_fit.blob_detection.minimize_blobiness.blob_operator import blob_operator


def minimal_representation(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, sigma: np.ndarray):
    # Check input for consistency.
    assert 0 < alpha < 1
    assert model.dim == m * n
    assert x_map.shape == (model.dim,)

    # Create the discretized operator \\nabla \\Delta_{x}^h G_h.
    dg = DiscreteGradient((m, n)).mat
    delta = DiscreteLaplacian((m, n), mode="reflect")
    basis = np.identity(m * n)
    out_list = []
    for basis_vector in basis:
        basis_image = np.reshape(basis_vector, (m, n))
        # Filter basis image.
        out = delta.fwd(gaussian(basis_image, sigma=sigma, mode="reflect").flatten())
        out_list.append(out)
    log_operator = np.column_stack(out_list)
    shape_operator = dg @ log_operator

    # Create the credible region
    cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)

    x_min = _solve_optimization_problem(a=shape_operator, c=cregion)

    # Reshape the optimizer into an image.
    f = np.reshape(x_min, (m, n))
    L_t = gaussian(image=f, sigma=sigma, mode="reflect")

    return L_t


BETA_REL = 1e-5


def _solve_optimization_problem(a: np.ndarray, c: CredibleRegion)\
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

    scale = np.sum(np.square(a @ c.x_map))
    alpha = 1 / scale
    beta = 0.0001

    problem = cp.Problem(cp.Minimize(alpha * cp.sum_squares(a @ x) + beta * cp.sum_squares(x)), constraints)

    # Solve the problem
    problem.solve(warm_start=False, verbose=False, solver=cp.ECOS)

    # Return the minimizer.
    x_min = x.value

    print(f"Blobiness of map: {np.sum(np.square(a @ c.x_map))}")
    print(f"Blobiness of solution: {np.sum(np.square(a @ x_min))}")
    return x_min
