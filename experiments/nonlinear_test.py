"""
Contains class "NonlinearTest"
"""

import numpy as np

from uq_mode import *

from model_fit import *
from experiments.experiment_kit import Test, TestResult


class NonlinearTest(Test):
    """
    Base class for tests with PixelModel as model.
    Child of class "Test".
    """
    def _initialize_model(self):
        y_noi = self.y_noi
        delta = self.sdev
        model = PixelModel(logname=self.outname, y=y_noi, standard_deviation=delta)
        # sample guess for theta_v
        q = 0.03    # 3% error
        theta_v_guess, regop2 = sample_theta(q=q, theta_v=self.theta_v)
        model.theta_v_bar = theta_v_guess
        model.regop2 = regop2
        model.set_starting_values(theta_v_start=theta_v_guess)
        return model

    def _change_model(self):
        raise NotImplementedError

    def _error_analysis(self, fitted_model: FittedPixelModel):
        costfun = fitted_model.rare_costfun
        error_analysis(y=self.y,
                       y_noi=self.y_noi,
                       f_true=self.f,
                       f_map=fitted_model.f_map,
                       theta_v_true=self.theta_v,
                       theta_v_map=fitted_model.theta_v_map,
                       costfun=costfun,
                       misfit=self.model.misfit)

    def _quantify_uncertainty(self, fitted_model: FittedPixelModel):
        # compute local credible intervals for both f and theta_v
        # make partition
        dim_f1 = self.model.dim_f1
        dim_f2 = self.model.dim_f2
        dim_f = self.model.dim_f
        dim_tv = self.theta_v.size
        # define 3x3 superpixels
        a_x = 3
        a_y = 3
        # get partition for the age-metallicity grid
        part = partition(n_x=dim_f1, n_y=dim_f2, a_x=a_x, a_y=a_y)
        # theta_v is treated as one whole partition element
        part.append(np.arange(dim_f, dim_f + dim_tv))
        # compute local credible intervals (actually local credible rectangles)
        xmap = np.concatenate((fitted_model.f_map, fitted_model.theta_v_map))
        # setup cost-dict for rci
        cost = {"fun": fitted_model.costfun, "grad": fitted_model.costgrad}
        intervals = rci(alpha=0.05, partition=part, n=dim_f + dim_tv, xmap=xmap, cost=cost,
                        lb=fitted_model.lb)
        return intervals

    def _plotting(self, fitted_model: FittedPixelModel, credible_intervals):
        f_true_image = np.reshape(self.f, (self.model.dim_f1, self.model.dim_f2))
        # setup uq-dictionary
        dimf = self.f.size
        if credible_intervals is not None:
            lci_f = credible_intervals[:dimf, :]
            lci_theta_v = credible_intervals[dimf:, :]
            uq = {"lci_f": lci_f, "lci_theta_v": lci_theta_v}
        else:
            uq = None
        plot_everything(savename=self.outname, f_true_im=f_true_image, f_map_im=fitted_model.f_map_image,
                        theta_v_true=self.theta_v, theta_v_map=fitted_model.theta_v_map, uq=uq)

    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        raise NotImplementedError

    def _rdm(self, f, theta_v):
        """
        Computes the relative data misfit of f
        :param f: ndarray
        :return: float
        """
        rdm = np.linalg.norm(self.model.misfit(f, theta_v)) / np.linalg.norm(self.model.misfit(self.f, self.theta_v))
        return rdm