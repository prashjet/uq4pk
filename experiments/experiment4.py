"""
Experiment 4: In this experiment we test the uncertainty quantification using local credible intervals based
on finding the smallest rectangle that contains the (localized) a-posteriori credible region.
We test the uncertainty quantification on the linear model using different regularization operators
(see experiment 1).
The quality of the uncertainty quantification is evaluated graphically by producing visualizations, and numerically
by computing quantitative error measures that indicate whether our approach to uncertainty quantification accurately
represents the "true" posterior uncertainty.
"""

import numpy as np

from uq4pk_fit.regop import *
from uq4pk_fit.uq_mode import is_credible

from uq4pk_fit.model_fit import *
from experiment_kit import Test, TestResult, SuperTest, TestSetup
from linear_test import LinearTest
from plotting_kit import *


class UQTestResult(TestResult):
    def __init__(self, snr, regop, truth_credible, e_rci, rci_size):
        TestResult.__init__(self)
        self.names = ["snr", "operator", "truth in CR", "LCI error", "LCI size"]
        self.attributes = [snr, regop, truth_credible, e_rci, rci_size]



class UQTest(LinearTest):

    def __init__(self, outname, f, setup):
        Test.__init__(self, outname, f, setup)
        self.ModelType = PixelModel
        self._alpha = 0.05   # credibility parameter

    def _change_model(self):
        regop = self.setup["regop"]
        if regop == "Identity":
            self.model.regop1 = TrivialOperator(dim=self.model.dim_f)
        elif regop == "Ornstein-Uhlenbeck":
            h = np.array([0.3, 2])
            self.model.regop1 = OrnsteinUhlenbeck(n1=self.model.dim_f1, n2=self.model.dim_f2, h=h)
        elif regop == "Gradient":
            self.model.regop1 = DiscreteGradient(n1=self.model.dim_f1, n2=self.model.dim_f2)
        elif regop == "Laplacian":
            self.model.regop1 = DiscreteLaplacian(n1=self.model.dim_f1, n2=self.model.dim_f2)
        else:
            raise Exception("Regop not recognized.")

    def _make_testresult(self, fitted_model: FittedLinearModel, credible_intervals) -> TestResult:
        fmap = fitted_model.f_map
        # check if truth lies in credible region
        ftrue = self.f.flatten()
        self.model.logger.activate()
        truth_credible = is_credible(x=ftrue, alpha=0.5, xmap=fmap, costfun=fitted_model.costfun)
        self.model.logger.deactivate()
        # evaluate errors
        err_lci = self._uq_error(credible_intervals)
        lci_ratios = np.divide(np.abs(credible_intervals[:,1]-credible_intervals[:,0]),
                               np.abs(self.f-fmap).clip(min=1))
        lci_size = np.mean(lci_ratios)
        result = UQTestResult(snr=self.setup["snr"],
                              regop=self.setup["regop"],
                              truth_credible=truth_credible,
                              e_rci=err_lci, rci_size=lci_size)
        return result

    def _uq_error(self, region):
        """
        Given a credible region [xmin, xmax], evaluates how much of the parameter xtrue=(f, theta_v)
        lies in [xmin, xmax].
        This is done by computing err_plus = (x - xmax)^+, err_minus = (x - xmin)^- and then
        err_uq = np.linalg.norm((err_plus; err_minus)).
        :return: float
            if 0, this means that x lies in [xmin, xmax]. Otherwise, indicates how much of x lies outside
            [xmin, xmax].
        """
        xmin = region[:, 0]
        xmax = region[:, 1]
        xtrue = self.f
        xtrue_too_large = xtrue - xmax
        xtrue_too_small = xmin - xtrue
        err_plus = self._positive_part(xtrue_too_large)
        err_minus = self._positive_part(xtrue_too_small)
        err_uq = np.linalg.norm(np.concatenate((err_plus, err_minus)), ord=1) / np.linalg.norm(xtrue, ord=1)
        return err_uq

    def _positive_part(self, x):
        """
        Returns the positive part of the vector x
        """
        return x.clip(min=0.)

    def _plot_uq(self, name, xi, vmax=None):
        """
        Plots the uncertainty quantification
        """
        # visualize uncertainty quantification for f
        ximin = np.reshape(xi[:, 0], (self.model._n_f_1, self.model._n_f_2))
        ximax = np.reshape(xi[:, 1], (self.model._n_f_1, self.model._n_f_2))
        # plot how much of the true is above ximax
        above = (self.f-ximax).clip(min=0.)
        below = (ximin-self.f).clip(min=0.)
        savename1 = f"{self.outname}/{name}_lower.png"
        savename2 = f"{self.outname}/{name}_upper.png"
        # determine vmax
        if vmax is None:
            vmax = np.max(xi.flatten())
        # plot the lower bound
        plot_with_colorbar(image=ximin, savename=savename1, vmax=vmax)
        plot_with_colorbar(image=ximax, savename=savename2, vmax=vmax)



class UQSuperTest(SuperTest):

    def __init__(self, output_directory, f_list):
        SuperTest.__init__(self, output_directory, f_list)
        self.ChildTest = UQTest

    def _setup_tests(self):
        setup_list = []
        #snr_list = [2000, 100]
        snr_list = [2000]
        regop_list = ["Identity", "Ornstein-Uhlenbeck", "Gradient", "Laplacian"]
        for snr in snr_list:
            for regop in regop_list:
                name = f"snr={snr}_{regop}"
                setup = TestSetup(name=name, parameters={"snr": snr, "theta_noise": 0., "regop": regop})
                setup_list.append(setup)
        return setup_list


# ------------------------------------------------------------------- RUN
name = "experiment4"
list_of_f = get_f("data", maxno=1)
super_test = UQSuperTest(output_directory=name, f_list=list_of_f)
super_test.perform_tests()