"""
Contains class "RMLSampler".
"""

from copy import deepcopy
from typing import List
from numpy.typing import ArrayLike

from ..external_packages import cgn
import numpy as np


class RMLSampler:
    def __init__(self, problem: cgn.Problem, starting_values: List[ArrayLike], solver_options: dict, reduction: float):
        """
        """
        # Change the regularization in the problem
        self._problem = deepcopy(problem)
        for param in self._problem._parameter_list:
            param.beta = param.beta / reduction
        self._starting_values = starting_values
        # Initialize solver
        self._solver = cgn.CGN()
        # Change solver options
        self._solver.options.tol = solver_options.setdefault("tol", self._solver.options.tol)
        self._solver.options.maxiter = solver_options.setdefault("maxiter", self._solver.options.maxiter)
        self._solver.options.set_verbosity(2)
        # Also need the root-covariance for creating the noise:
        self._qinv = np.linalg.inv(self._problem.q.a)

    def sample(self, n):
        """
        Creates "approximate" samples form the posterior distribution using the randomized maximum likelihood-method.
        :param n: int
            The desired number of samples.
        :return: array_like, shape (K,n)
            Returns the samples as an array of shape (K,n), where K is the dimension of the parameter space, and each
            column corresponds to a sample.
        """
        m = self._problem.m
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
            std_noise = np.random.randn(m)
            noise = self._qinv @ std_noise
            # Add the noise to the "true" measurement to obtain a perturbed measurement y_i.
            x_list = self._fit_model(noise)
            # Concatenate x_map_list to a numpy vector.
            x = np.concatenate(x_list)
            # Append to the list of samples.
            sample_list.append(x)
        # Turn the list of samples into a numpy array of the desired shape.
        sample_arr = np.column_stack(sample_list)
        return sample_arr

    def _fit_model(self, noise):
        # Change misfit to account for added noise.
        old_fun = deepcopy(self._problem.fun)
        def new_func(*args):
            f = old_fun(*args)
            return f + noise
        self._problem.fun = new_func
        # Solve perturbed optimization problem with CGN
        optsol = self._solver.solve(problem=self._problem, starting_values=self._starting_values)
        x_map_list = optsol.minimizer
        # Return the solution
        return x_map_list
