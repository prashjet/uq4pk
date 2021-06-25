"""
Contains class 'Normalized PixelModel'
"""

import numpy as np

from uq4pk_fit.cgn import *

from .model import Model, FittedModel
from uq4pk_fit.regop import OrnsteinUhlenbeck


class NormalizedPixelModel(Model):
    """
    Implements a basic pixel-by-pixel model with standard l2-regularization AND sum-to-one constraints.
    The statistical model is:
        y = G(f, theta_v) + noise,
        noise ~ normal(0, identity / delta^2),
        f ~ normal(f_bar, cov1),   where cov1 = (alpha1 * regop1 @ regop1.T)^(-1)
        theta_v ~ normal(theta_v_bar, cov2),   where co2 = (alpha2 * regop2 @ regop2.T)^(-1),
        sum(f) = 1, f >= 0.
    """
    def __init__(self, y, standard_deviation, logname="pixel_model"):
        Model.__init__(self, y, logname)
        self.logger.activate()
        self.delta = standard_deviation
        self._set_solver_options()
        self.misfit = self._misfit_handler.misfit
        self.misfitjac = self._misfit_handler.misfitjac
        # setup constraint
        c = np.ones((1, self.dim_f))
        d = np.ones(1).reshape((1,))
        self.equality_constraint = {"mat": c, "vec": d}
        self.lb = np.zeros(self.dim_f)
        print(f"Normalized pixel model successfully initialized.")
        self.logger.deactivate()

    # PRIVATE

    def _set_regularization_parameters(self):
        """
        Sets the default regularization parameters
        """
        # set regularization parameters for f
        self.alpha1 = 0.1
        self.f_bar = np.zeros(self.dim_f)
        h = np.array([0.3, 2])
        self.regop1 = OrnsteinUhlenbeck(n1=self.dim_f1, n2=self.dim_f2, h=h)
        # set regularization parameters for theta_v
        self.alpha2 = 1.
        self.theta_v_bar = np.array([20, 90, 0., 0., 0., 0., 0.])
        std_theta_v = np.array([10., 10., 1., 1., 1., 1., 1.])
        self.regop2 = DiagonalOperator(dim=self.dim_theta_v, s=np.divide(1, std_theta_v))
        # scale with 1/m to get nicer numbers
        self.scaling = 1 / self._m
        # default starting values
        self.f_start = np.zeros(self.dim_f)
        self.f_start_flag = True
        self.theta_v_start = self.theta_v_bar
        self.theta_v_start_flag = True

    def _setup_solver(self):
        # create an object of type MultiParameterProblem
        pars = Parameters()

        pars.addParameter(dim=self.dim_f, mean=self.f_bar, alpha=self.alpha1, regop=self.regop1,
                          eq=self._equality_constraint, lb=self.lb)
        pars.addParameter(dim=self.dim_theta_v, mean=self.theta_v_bar, alpha=self.alpha2, regop=self.regop2)
        problem = MultiParameterProblem(pars, fun=self.misfit, jac=self.misfitjac, delta=self.delta,
                                        scaling=self.scaling)
        # create a BGN object
        solver = CGN(problem=problem)
        return solver

    def _set_solver_options(self):
        """
        Overwrites solveroptions from super class "Model"
        """
        self.solveroptions = {"maxiter": 200, "timeout": 60, "verbose": True, "tol": 0.01, "gtol": 1e-16, "lsiter": 66}

    def _assemble_fitted_model(self, xmap, precision, costfun, info):
        fitted_model = FittedNormalizedPixelModel(model=self, xmap=xmap, precision=precision, costfun=costfun,
                                        info=info)
        return fitted_model


class FittedNormalizedPixelModel(FittedModel):
    """
    Represents the fitted model, i.e. the PixelModel linearized around the map:
        y = G(f_map, theta_v_map) + G'(f_map, theta_v_map)[f-f_map, theta_v-theta_v_map] + noise,
        noise ~ normal(0, identity / delta^2),
        f ~ normal(f_bar, cov1),   where cov1 = (alpha1 * regop1 @ regop1.T)^(-1)
        theta_v ~ normal(theta_v_bar, cov2),   where co2 = (alpha2 * regop2 @ regop2.T)^(-1),
        sum(f)=1, f >= 0.
    """
    def __init__(self, model: NormalizedPixelModel, xmap, precision, costfun, info):
        FittedModel.__init__(self, xmap, precision, costfun, info)
        self._nf = model.dim_f
        self._ntheta_v = model.dim_theta_v
        self._n = self._nf + self._ntheta_v
        self.f_map = xmap[0]
        self.theta_v_map = xmap[1]
        self.xmap = np.concatenate((xmap[0], xmap[1]))
        self._misfit_map = model.misfit(self.f_map, self.theta_v_map)
        self._misfitjac_map = model.misfitjac(self.f_map, self.theta_v_map)
        self.delta = model.delta
        self.regop1 = model.regop1
        self.regop2 = model.regop2
        self.alpha1 = model.alpha1
        self.alpha2 = model.alpha2
        self.f_bar = model.f_bar
        self.theta_v_bar = model.theta_v_bar
        # for the pixel model, the lower bound is 0 for f and -inf for theta_v
        self.lb = - np.inf * np.ones(self._n)
        self.lb[:self._nf] = 0
        self._equality_constraint = model.equality_constraint
        # also want f_map as an image for easier handling
        self.f_map_image = np.reshape(self.f_map, (model.dim_f1, model.dim_f2))

    def _split(self, x):
        f = x[:self._nf]
        theta = x[self._nf]
        return f, theta

    def costfun(self, x):
        mf = self._mfterm(x)
        reg = self._regterm(x)
        return mf + reg

    def costgrad(self, x):
        mfgrad = self._mfterm_grad(x)
        reggrad = self._regterm_grad(x)
        return mfgrad + reggrad

    def _misfit(self, f, tv):
        x = np.concatenate((f, tv))
        return self._misfit_map + self._misfitjac_map @ (x-self.xmap)

    def _misfitjac(self, f, tv):
        return self._misfitjac_map

    def _mfterm(self, x):
        value = 0.5 * (np.linalg.norm(self._misfit_map + self._misfitjac_map @ (x-self.xmap)) / self.delta) ** 2
        return value

    def _mfterm_grad(self, x):
        grad = self._misfitjac_map.T @ (self._misfit_map + self._misfitjac_map @ (x-self.xmap)) / (self.delta ** 2)
        return grad

    def _regterm(self, x):
        f, theta = self._split(x)
        reg_f = 0.5 * self.alpha1 * np.linalg.norm(self.regop1.fwd(f-self.f_bar))**2
        reg_theta = 0.5 * self.alpha2 * np.linalg.norm(self.regop2.fwd(theta-self.theta_v_bar))
        return reg_f + reg_theta

    def _regterm_grad(self, x):
        f, theta = self._split(x)
        reg_f_df = self.alpha1 * self.regop1.mat.T @ self.regop1.fwd(f-self.f_bar)
        reg_f_grad = np.concatenate((reg_f_df, np.zeros(self._ntheta_v)))
        reg_theta_dtheta = self.alpha2 * self.regop2.mat.T @ self.regop2.fwd(theta - self.theta_v_bar)
        reg_theta_grad = np.concatenate((np.zeros(self._nf), reg_theta_dtheta))
        return reg_f_grad + reg_theta_grad
