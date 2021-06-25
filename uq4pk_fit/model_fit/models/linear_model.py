"""
Contains class "LinearModel"
"""

import numpy as np

from uq4pk_fit.cgn import *

from uq4pk_fit.model_fit.models.model import Model, FittedModel
from uq4pk_fit.regop import OrnsteinUhlenbeck


class LinearModel(Model):
    """
    Implements a basic linear model with theta_v fixed.
    This corresponds to the linear statistical model:
        y = G(f,theta_v) + noise,
        noise ~ normal(0, identity / delta^2),
        f ~ normal(f_bar, cov1),   where cov1 = (alpha1 * regop1 @ regop1.T)^(-1),
        theta_v given.
    """
    def __init__(self, y, standard_deviation, theta_v, logname="pixel_model"):
        Model.__init__(self, y, logname)
        self.logger.activate()
        self.theta_v = theta_v
        self.delta = standard_deviation
        print(f"delta = {self.delta}")
        self._set_solver_options()
        # setup different misfit and misfitjac
        print(f"Linear model successfully initialized.")
        self.logger.deactivate()

    def misfit(self, f):
        rm = self._misfit_handler.misfit(f, self.theta_v)
        return rm

    def misfitjac(self, f):
        return self._misfit_handler.misfitjac(f, self.theta_v)[:, :self.dim_f]

    # PRIVATE

    def _set_regularization_parameters(self):
        """
        Sets the default regularization parameters
        """
        # set regularization parameters for f
        self.alpha1 = 1.
        self.f_bar = np.zeros(self.dim_f)
        h = np.array([0.3, 2])
        self.regop1 = OrnsteinUhlenbeck(n1=self.dim_f1, n2=self.dim_f2, h = h)

        self.scaling = 1 / self._m

        self.f_start = np.zeros(self.dim_f)
        self.f_start_flag = True

    def _setup_solver(self):
        # create an object of type MultiParameterProblem
        pars = Parameters()
        lb = np.zeros(self.dim_f)
        pars.addParameter(dim=self.dim_f, mean=self.f_bar, alpha=self.alpha1, regop=self.regop1, lb=lb)
        problem = MultiParameterProblem(pars, fun=self.misfit, jac=self.misfitjac, delta=self.delta,
                                        scaling=self.scaling)
        # create a BGN object
        solver = CGN(problem=problem)
        return solver

    def _set_solver_options(self):
        """
        Overwrites solveroptions from super class "Model"
        """
        self.solveroptions = {"maxiter": 200, "timeout": 60, "verbose": True, "tol": 0.01, "gtol": 0.1, "lsiter": 66}

    def _assemble_fitted_model(self, xmap, precision, costfun, info):
        fitted_model = FittedLinearModel(model=self, xmap=xmap, precision=precision, costfun=costfun, info=info)
        return fitted_model


class FittedLinearModel(FittedModel):
    """
    Represents the fitted model, i.e. the PixelModel linearized around the map:
        y = G(f_map, theta_v_map) + G'(f_map, theta_v_map)[f-f_map, theta_v-theta_v_map] + noise,
        noise ~ normal(0, identity / delta^2),
        f ~ normal(f_bar, cov1),   where cov1 = (alpha1 * regop1 @ regop1.T)^(-1)
        theta_v ~ normal(theta_v_bar, cov2),   where co2 = (alpha2 * regop2 @ regop2.T)^(-1),
        f >= 0.
    """
    def __init__(self, model: LinearModel, xmap, precision, costfun, info):
        FittedModel.__init__(self, xmap=xmap, precision=precision, costfun=costfun, info=info)
        self.f_map = xmap
        self.f_map_image = np.reshape(self.f_map, (model.dim_f1, model.dim_f2))
        self.theta_v = model.theta_v
        # for the linear model, the lower bound is 0
        self.lb = np.zeros(model.dim_f)
        self._mf_map = model.misfit(self.f_map)
        self._mfjac_map = model.misfitjac(self.f_map)
        self._nf = model.dim_f
        self._ntheta_v = model.dim_theta_v
        self.delta = model.delta
        self.regop1 = model.regop1
        self.alpha1 = model.alpha1
        self.f_bar = model.f_bar
        self._n = self._nf
        self.info = info

    def costfun(self, f):
        mf = self._mfterm(f)
        reg = self._regterm(f)
        return mf + reg

    def costgrad(self, f):
        mfgrad = self._mfterm_grad(f)
        reggrad = self._regterm_grad(f)
        return mfgrad + reggrad

    def _mfterm(self, f):
        w = f - self.f_map
        value = 0.5 * (np.linalg.norm(self._mf_map + self._mfjac_map @ w) / self.delta) ** 2
        return value

    def _mfterm_grad(self, f):
        w = f - self.f_map
        grad = self._mfjac_map.T @ (self._mf_map + + self._mfjac_map @ w) / (self.delta ** 2)
        return grad[:self._nf]

    def _regterm(self, f):
        reg_f = 0.5 * self.alpha1 * np.linalg.norm(self.regop1.fwd(f-self.f_bar))**2
        return reg_f

    def _regterm_grad(self, f):
        reg_f_df = self.alpha1 * self.regop1.mat.T @ self.regop1.fwd(f-self.f_bar)
        return reg_f_df
