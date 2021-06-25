"""
Contains class "LinearTest"
"""

import numpy as np

from uq4pk_fit.uq_mode import *

from uq4pk_fit.model_fit import *
from experiments.experiment_kit import Test, TestResult
from plotting_kit import *


class LinearTest(Test):
    """
    Base class for tests with LinearModel as model.
    Child of class "Test".
    """
    def _initialize_model(self):
        y_noi = self.y_noi
        delta = self.sdev
        theta_v = self.theta_v
        model = LinearModel(logname=self.outname, y=y_noi, theta_v=theta_v, standard_deviation=delta)
        return model

    def _error_analysis(self, fitted_model: FittedLinearModel):
        # setup costfunction and misfit function so that they have the right format
        costfun = fitted_model.rare_costfun
        misfit = self.model.misfit
        error_analysis(y=self.y,
                       y_noi=self.y_noi,
                       f_true=self.f,
                       f_map=fitted_model.f_map,
                       theta_v_true=self.theta_v,
                       costfun=costfun,
                       misfit=misfit)

    def _plotting(self, fitted_model: FittedLinearModel, credible_intervals):
        f_true_image = np.reshape(self.f, (self.model.dim_f1, self.model.dim_f2))
        # setup uq-dictionar
        if credible_intervals is not None:
            uq = {"lci_f": credible_intervals}
        else:
            uq = None
        plot_everything(savename=self.outname, f_true_im=f_true_image, f_map_im=fitted_model.f_map_image,
                        theta_v_true=self.theta_v, uq=uq)

    def _quantify_uncertainty(self, fitted_model: FittedLinearModel):
        # compute local credible intervals for both f and theta_v
        # make partition
        dim_f1 = self.model.dim_f1
        dim_f2 = self.model.dim_f2
        dim_f = self.model.dim_f
        # define superpixel with side length 3
        a_x = 3
        a_y = 3
        # get partition for the age-metallicity grid
        part = partition(n_x=dim_f1, n_y=dim_f2, a_x=a_x, a_y=a_y)
        # compute local credible intervals (actually local credible rectangles)
        xmap = fitted_model.f_map
        # setup cost-dict
        # test costfunction
        cost = {"fun": fitted_model.costfun, "grad": fitted_model.costgrad}
        intervals = rci(alpha=0.05, partition=part, n=dim_f, xmap=xmap, cost=cost,
                        lb=fitted_model.lb)
        return intervals

    def _rdm(self, f):
        """
        Computes the relative data misfit of f
        :param f: ndarray
        :return: float
        """
        rdm = np.linalg.norm(self.model.misfit(f)) / np.linalg.norm(self.model.misfit(self.f))
        return rdm

    def _change_model(self):
        raise NotImplementedError

    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        raise NotImplementedError
