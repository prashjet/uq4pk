"""
Contains classes "FittedModel" and "Model"
"""

import os

from .logger import Logger
from .misfit_handler import MisfitHandler


class FittedModel:
    """
    Contains all the stuff that one might be interested in after fitting a model.
    """
    def __init__(self, xmap, precision, costfun, info=None):
        self.precision = precision
        self.info = info
        self.rare_costfun = costfun

    def costfun(self, x):
        raise NotImplementedError

    def costgrad(self, x):
        raise NotImplementedError


class Model:
    """
    Abstract base class for that manages the optimization problem, regularization, and optionally also the
    uncertainty quantification.
    """
    def __init__(self, y, logname="model"):
        # copy string
        output_directory = '%s' % logname
        os.makedirs(output_directory, exist_ok=True)
        self.logger = Logger(f"{output_directory}_logfile.log")
        # setup misfit handler
        self._misfit_handler = MisfitHandler(y=y, hermite_order=4)
        # get parameter dimensions from misfit handler
        self.dim_f1, self.dim_f2, self.dim_theta_v = self._misfit_handler.get_dims()
        self.dim_f = self.dim_f1 * self.dim_f2
        self._m = y.size
        # set regularization parameters (default, user can change them afterwards)
        self._set_regularization_parameters()
        # default solveroptions
        self.solveroptions = {"maxiter": 100, "timeout": 60, "verbose": True, "tol": 1e-6, "gtol": 1e-6,
                              "ctol": 1., "lsiter": 100}
        self._f_start_flag = False
        self._theta_v_start_flag = False
        self._f_start = None
        self._theta_v_start = None
        self._solver = None

    def set_starting_values(self, f_start=None, theta_v_start=None):
        """
        Set the starting values for the optimization
        :param f_start: starting value for the distribution function f (in image format, or not)
        :param theta_v_start: starting value for theta_v
        """
        if f_start is not None:
            self._f_start = f_start.flatten()
            self._f_start_flag = True
        if theta_v_start is not None:
            self._theta_v_start = theta_v_start
            self._theta_v_start_flag = True

    def fit(self) -> FittedModel:
        """
        Runs the optimization to fit the model.
        :return: FittedModel
        """
        self.logger.activate()
        print("Setting up the _solver.")
        self._solver = self._setup_solver()
        if self._f_start_flag:
            self._solver.set_starting_value(0, self._f_start)
        if self._theta_v_start_flag:
            self._solver.set_starting_value(1, self._theta_v_start)
        sol = self._solver.solve(self.solveroptions)
        xmap = sol.minimizer
        precision = sol.precision
        info = sol.info
        costfun = sol.costfun
        # setup the linearized costfun
        fitted_model = self._assemble_fitted_model(xmap=xmap, precision=precision, costfun=costfun, info=info)
        self.logger.deactivate()
        return fitted_model

    # PRIVATE

    def _set_regularization_parameters(self):
        raise NotImplementedError

    def _setup_solver(self):
        raise NotImplementedError

    def _assemble_fitted_model(self, xmap, precision, costfun, info):
        raise NotImplementedError
