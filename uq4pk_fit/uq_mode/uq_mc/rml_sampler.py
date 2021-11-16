import numpy as np

import uq4pk_fit.cgn as cgn
from ..linear_model import LinearModel


class RMLSampler:
    """
    Manages the generation of samples from the perturbed statistical model.
    """
    def __init__(self, model: LinearModel, x_map: np.ndarray, options: dict):
        """
        """
        self._model = model
        self._x_map = x_map
        # Get reduction factor
        self._reduction = options.setdefault("reduction", 10.)
        # Initialize CGN solver
        self._solver = cgn.CGN()
        self._solver.options.tol = options.setdefault("tol", self._solver.options.tol)
        self._solver.options.maxiter = options.setdefault("maxiter", self._solver.options.maxiter)
        self._solver.options.set_verbosity(0)
        # Also need the root-covariance for creating the noise:
        self._qinv = np.linalg.inv(model.q.mat)

    def sample(self, n: int):
        """
        Creates "approximate" samples form the posterior distribution using the randomized maximum likelihood-method.
        :param n: The desired number of samples.
        :returns: The samples as an array of shape (K,n), where K is the dimension of the parameter space, and each
            column corresponds to a sample.
        """
        ydim = self._model.ydim
        # Initialize the list where the samples will be stored.
        sample_list = []
        print(" ")
        for i in range(n):
            print("\r", end="")
            print(f"Computing sample {i+1}/{n}", end=" ")
            # Sample x_i from the prior.
            # x_bar_list = self._model.mean_list
            # regop_list = self._model.regop_list
            # x_i_list = []
            # for x_bar, regop in zip(x_bar_list, regop_list):
            #     std_noise = np.random.randn(regop.rdim)
            #     noise = regop.inv(std_noise)
            #     x_i = x_bar + noise
            #     x_i_list.append(x_i)
            # Create noise that has the right covariance.
            std_noise = np.random.randn(ydim)
            noise = self._qinv @ std_noise
            # Add the noise to the "true" measurement to obtain a perturbed measurement y_i.
            x = self._fit_model(noise)
            # Append to the list of samples.
            sample_list.append(x)
        # Turn the list of samples into a numpy array of the desired shape.
        sample_arr = np.column_stack(sample_list)
        return sample_arr

    def _fit_model(self, noise: np.ndarray) -> np.ndarray:
        """
        Computes the MAP estimate of the perturbed statistical model.
        :param noise: The perturbation noise.
        :return:
        """
        # Make problem object from model
        x = cgn.Parameter(dim=self._model.n, name="x")
        x.lb = self._model.lb
        x.beta = 1 / self._reduction
        # Change misfit to account for added noise.
        def perturbed_misfit(x):
            f = self._model.h @ x - self._model.y
            return f + noise
        def jac(x):
            return self._model.h
        perturbed_problem = cgn.Problem(parameters=[x], fun=perturbed_misfit,
                                        jac=jac, q=self._model.q, scale=self._model.ydim)
        if self._model.a is not None:
            eqcon = cgn.LinearConstraint(parameters=[x], a=self._model.a, b=self._model.b, ctype="eq")
            perturbed_problem.constraints.append(eqcon)
        # Solve perturbed optimization problem with CGN
        optsol = self._solver.solve(problem=perturbed_problem, starting_values=[self._x_map])
        x = optsol.minimizer("x")
        # Return the solution
        return x
