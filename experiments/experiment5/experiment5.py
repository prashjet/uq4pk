"""
Experiment 5: A second test for the uncertainty quantification, but this time for the nonlinear inference. This means
we also obtain local credible intervals for theta_v, which we evaluate graphically and numerically.
"""

from uq4pk_fit.inference import *

from experiments.experiment_kit import *


class Experiment5Result(TrialResult):
    def _compute_results(self):
        names = ["uqerrorf", "uqtightnessf", "uqerrortheta", "uqtightnesstheta"]
        values = [self.uqerr_f, self.uqtightness_f, self.uqerr_theta, self.uqtightness_theta]
        return names, values

    def _additional_plotting(self, savename):
        pass


class Experiment5Trial(Trial):

    def _choose_test_result(self):
        return Experiment5Result

    def _change_model(self):
        pass

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci")
        return uq


class Experiment5(Experiment):

    def _set_child_test(self):
        return Experiment5Trial

    def _setup_tests(self):
        setup_list = []
        setup = TestSetup({})
        setup_list.append(setup)
        return setup_list
