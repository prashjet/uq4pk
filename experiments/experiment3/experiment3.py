"""
Experiment 3: In this experiment, we test how sensitive the reconstruction is to the regularization parameter beta1.
To this end, we fit the nonlinear inference with an (again noisy) initial guess for theta_v but different
values for the regularization parameter beta1.
"""

from uq4pk_fit.inference import *

from experiments.experiment_kit import *


class Experiment3Result(TrialResult):

    def _compute_results(self):
        names = ["errrof", "ssimf", "errortheta"]
        err_f = self.err_f
        ssimf = self.ssim_f
        err_theta_v = self.sre_tv
        attributes = [err_f, ssimf, err_theta_v]
        return names, attributes

    def _additional_plotting(self, savename):
        pass


class Experiment3Trial(Trial):

    def _choose_test_result(self):
        return Experiment3Result

    def _change_model(self):
        self.model.normalize()
        self.model.beta2 = self.setup.parameters["beta2"]

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Experiment3(Experiment):

    def _set_child_test(self):
        return Experiment3Trial

    def _setup_tests(self):
        setup_list = []
        beta2_list = [0.1, 1., 10., 100, 1000]
        for beta2 in beta2_list:
            setup = TestSetup({"beta2": beta2})
            setup_list.append(setup)
        return setup_list
