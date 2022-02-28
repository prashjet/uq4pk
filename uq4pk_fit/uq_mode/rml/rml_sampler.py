import numpy as np

import uq4pk_fit.cgn as cgn
from ..linear_model import LinearModel


class RMLSampler:
    """
    Manages the generation of samples from the perturbed statistical model.
    """
    def __init__(self, model: LinearModel, x_map: np.ndarray, sample_prior: bool, options: dict):
        """
        """
        self._sample_prior = sample_prior
        self._model = model
        self._x_map = x_map
        # Get reduction factor
        self._reduction = options.setdefault("reduction", 1.)
        # Initialize CGN solver
        self._solver = cgn.CGN()
        self._solver.options.tol = options.setdefault("tol", self._solver.options.tol)
        self._solver.options.maxiter = options.setdefault("maxiter", self._solver.options.maxiter)
        self._solver.options.set_verbosity(2)

    def sample(self, n: int):
        """
        Creates "approximate" samples form the posterior distribution using the randomized maximum likelihood-method.
        :param n: The desired number of samples.
        :returns: The samples as an array of shape (dim, d), where d is the dimension of the parameter space, and each
            column corresponds to a sample.
        """
        ydim = self._model.ydim
        # Initialize the list where the samples will be stored.
        sample_list = []
        print(" ")
        for i in range(n):
            print("\r", end="")
            print(f"Computing sample {i+1}/{n}", end=" ")
            # SET REGULARIZING GUESS
            if self._sample_prior:
                # Sample x_bar from the prior.
                # Create standard normal noise.
                std_noise = np.random.randn(self._model.dim)
                # R prior_noise = std_noise <=> prior_noise ~ normal(0, (RR.T)^(-1)).
                prior_noise = self._model.r.inv(std_noise)
                # Add noise to the prior mean
                x_bar = self._model.m + prior_noise
            else:
                # Regularizing guess is equal to prior mean.
                x_bar = self._model.m

            # CREATE PERTURBATION OF MEASUREMENT
            std_noise = np.random.randn(ydim)
            noise = np.linalg.solve(self._model.q.inv(std_noise))
            # Add the noise to the "true" measurement to obtain a perturbed measurement y_i.
            x = self._fit_model(x_bar, noise)
            # Append to the list of samples.
            sample_list.append(x)
        # Turn the list of samples into a numpy array of the desired shape.
        sample_arr = np.row_stack(sample_list)

        assert sample_arr.shape == (n, self._model.dim)
        return sample_arr

    def _fit_model(self, x_bar: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Computes the MAP estimate of the perturbed statistical model.
        :param x_bar: Of shape (d, ). The regularizing guess.
        :param noise: Of shape (m, ). The perturbation noise.
        :return: The maximum-a-posteriori estimate for the perturbed model. Of shape (d,).
        """
        # Check input
        assert x_bar.shape == (self._model.dim,)
        assert noise.shape == (self._model.ydim, )

        # Make problem object from model
        x = cgn.Parameter(dim=self._model.dim, name="x")
        x.mean = x_bar
        x.lb = self._model.lb
        x.beta = 1. / self._reduction
        x.regop = self._model.r
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
        assert x.shape == (self._model.dim, )
        return x
