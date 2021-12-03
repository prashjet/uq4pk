
from cyipopt import minimize_ipopt
from math import log, sqrt
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as sciopt
from skimage import morphology
from typing import List

import uq4pk_fit.cgn as cgn
from uq4pk_fit.special_operators import DiscreteGradient
from uq4pk_fit.uq_mode.linear_model import LinearModel
from .blob_operator import blob_operator
from .scale_space_representation import scale_space_representation
from .plotty_blobby import plotty_blobby


def bump_minimization2(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray,min_scale: float,
                       max_scale: float) -> np.ndarray:
    """
    Computes the minimizer of the optimization problem

    .. math::
        min_f ||\\nabla \\Delta_{x}^h L||_2^2 + \\beta ||f||_2^2
        s.t. L(x, h) = G_h f, f \\in C_{\\alpha},

    where :math:`G_h` is the isotropic Gaussian filter with scale parameter h.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The linear statistical model.
    :param x_map: The MAP estimate for ``model``.
    :param min_scale: The minimum scale for the scale-space representation.
    :param max_scale: The maximum scale for the scale-space representation.
    :return: Numpy array of shape (m, n).
    """
    # Check input for consistency.
    _check_input(alpha, m, n, model, x_map, min_scale, max_scale)
    # Create the scale-discretization.
    scales = _discretize_scales(min_scale, max_scale)
    # Create the discretized operator \\nabla \\Delta_{x}^h G_h.
    print("Creating blob operator...")
    dg = DiscreteGradient((len(scales), m, n)).mat
    shape_operator = dg @ blob_operator(scales, m, n)
    #shape_operator = blob_operator(scales, m, n)    # For debugging purposes.
    # Set up the optimization problem.
    print("Setting up CGN problem...")
    problem = _setup_cgn_problem(alpha=alpha, x_map=x_map, model=model, shape_operator=shape_operator)
    # Solve the optimization problem with CGN.
    solver = cgn.CGN()
    solver.options.set_verbosity(lvl=2)
    solver.options.ctol = 1e-2
    solver.linesearch.maxviol = 1000.  # Do not allow more than 100 % constraint violation.
    solver.options.maxiter = 500
    print("Starting to solve...")
    problem_solution = solver.solve(problem=problem, starting_values=[x_map])
    x_min = problem_solution.minimizer("x")
    # Reshape the optimizer into an image and then return it.
    f = np.reshape(x_min, (m, n))
    return f


def bump_minimization_ipopt(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray,min_scale: float,
                       max_scale: float, num_scale=6) -> np.ndarray:
    """
    Computes the minimizer of the optimization problem

    .. math::
        min_f ||\\nabla \\Delta_{x}^h L||_2^2 + \\beta ||f||_2^2
        s.t. L(x, h) = G_h f, f \\in C_{\\alpha},

    where :math:`G_h` is the isotropic Gaussian filter with scale parameter h.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The linear statistical model.
    :param x_map: The MAP estimate for ``model``.
    :param min_scale: The minimum scale for the scale-space representation.
    :param max_scale: The maximum scale for the scale-space representation.
    :return: Numpy array of shape (m, n).
    """
    # Check input for consistency.
    _check_input(alpha, m, n, model, x_map, min_scale, max_scale)
    # Create the scale-discretization.
    scales = _discretize_scales(min_scale, max_scale, num_scale)
    # Create the discretized operator \\nabla \\Delta_{x}^h G_h.
    print("Creating blob operator...")
    dg = DiscreteGradient((len(scales), m, n)).mat
    blobby = blob_operator(scales, m, n)
    shape_operator = dg @ blobby
    # Solve the resulting problem with IPOPT.
    print("Setting up IPOPT problem...")
    problem = _setup_cgn_problem(alpha=alpha, x_map=x_map, model=model, shape_operator=shape_operator)
    # Solve the optimization problem with CGN.
    inequality_constraint = {"fun": problem.constraints[0].fun, "jac": problem.constraints[0].jac, "type": "ineq"}
    # Quick and dirty
    lb = model.lb
    ub = np.inf * np.ones(m * n)
    bnds = np.column_stack([lb, ub])
    x_min = minimize_ipopt(fun=problem.costfun, jac=problem.costgrad, x0=x_map, constraints=(inequality_constraint),
                           bounds=bnds, options={"maxiter": 5000}).x
    # Reshape the optimizer into an image and then return it.
    f = np.reshape(x_min, (m, n))
    # But first, plot the scale-space representation of the solution.
    f_ssr = scale_space_representation(f, scales)
    # Finally, plot the scale-slices.
    # And determine local minima of Delta^h_x L
    delta_h_l = blobby @ x_min
    delta_h_l = np.reshape(delta_h_l, (len(scales), m, n))
    i = 0
    for l in delta_h_l:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(l, cmap="gnuplot")
        i += 1
    plt.show()
    # And determine local minima of Delta^h_x L
    blobs = morphology.local_minima(delta_h_l, indices=True, allow_borders=True)
    blobs = np.array(blobs).T
    # Remove local minima above threshold
    rthresh = 0.1
    athresh = rthresh * abs(delta_h_l.min())
    large_enuf = []
    for i in range(blobs.shape[0]):
        blob_i = blobs[i].flatten()
        if abs(delta_h_l[tuple(blob_i)]) >= athresh:
            large_enuf.append(i)
    blobs = blobs[large_enuf]
    blobs = [blob for blob in blobs]
    # Show MAP estimate with blobs
    plotty_blobby(np.reshape(x_map, (m, n)), blobs=blobs, scales=scales)
    plt.show()
    return f


def _check_input(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, min_scale: float,
                 max_scale: float):
    assert 0 < alpha < 1
    assert model.n == m * n
    assert x_map.shape == (model.n, )
    assert 0 < min_scale <= max_scale


def _discretize_scales(min_scale: float, max_scale: float, num_scale: int) -> List[float]:
    """
    Creates a scale-discretization of the form.

    :param min_scale:
    :param max_scale:
    :return:
    """
    assert min_scale <= max_scale
    scale_step = (max_scale - min_scale) / num_scale
    scales = [min_scale + n * scale_step for n in range(num_scale + 1)]
    return scales


BETA_REL = 1e-5


def _setup_cgn_problem(alpha: float, x_map: np.ndarray, model: LinearModel, shape_operator: np.ndarray) -> cgn.Problem:
    """
    Creates the CGN problem

    .. math::
        min_f ||Q f||_2^2 + \\beta ||f||_2^2 s.t. c(f) >= 0, A f = b,
    where c_{\\alpha} is the Pereyra constraint function.

    :param alpha:
    :param x_map:
    :param model:
    :param shape_operator:
    :return: The optimization problem as :py:class:`cgn.Problem` object.
    """
    # Initialize parameter with possible bound constraints.
    n = x_map.size
    x = cgn.Parameter(dim=n, name="x")
    if model.lb is not None:
        x.lb = model.lb
    # Add a regularization term and set the regularization parameter.
    # We want that \\beta ||f - f_map||_2^2 is much smaller than ||Q f||_2^2
    loss = 0.5 * np.sum(np.square(shape_operator @ x_map))
    reg = 0.5 * np.sum(np.square(x_map))
    x.beta = BETA_REL * loss / reg
    # Setup the misfit function and its Jacobian.

    def misfit(z):
        return shape_operator @ z

    def misfit_jac(z):
        return shape_operator
    # Setup the Pereyra constraint.
    tau_alpha = sqrt(16 * log(3 / alpha) / n)
    k_alpha = n * (tau_alpha + 1)
    cost_map = model.cost(x_map)
    inscale = 1 / k_alpha

    def incon(z):
        c = cost_map + k_alpha - model.cost(z)
        return inscale * c * np.ones((1, ))

    def incon_jac(z):
        return - inscale * model.cost_grad(z).reshape((1, n))
    inequality_constraint = cgn.NonlinearConstraint(parameters=[x], fun=incon, jac=incon_jac, ctype="ineq")
    constraints = [inequality_constraint]
    # Optionally also setup the equality constraint.
    if model.a is not None:
        eqcon = cgn.LinearConstraint(parameters=[x], a=model.a, b=model.b, ctype="eq")
        constraints.append(eqcon)
    # Create the cgn.Problem object and return it.
    print(f"Scale = {loss}")
    cgn_problem = cgn.Problem(parameters=[x], fun=misfit, jac=misfit_jac, constraints=constraints, scale=loss)
    return cgn_problem
