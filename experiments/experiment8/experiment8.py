"""
This experiment compares the filtered credible intervals computed via the Pereyra approximation to the ones
computed via RML (as a baseline).
The basic model is the nonlinear one, with theta_v not fixed at all.
"""

import numpy as np

from uq4pk_fit.inference import *
from experiments.experiment_kit import *


class Experiment8Result(TrialResult):
    def _compute_results(self):
        names = ["uqerrorf", "uqtightnessf", "uqerrortheta", "uqtightnesstheta"]
        uq_error_f = self.uqerr_f
        uq_tightness_f = self.uqtightness_f
        uq_error_theta = self.uqerr_theta
        uq_tightness_theta = self.uqtightness_theta
        values = [uq_error_f, uq_tightness_f, uq_error_theta, uq_tightness_theta]
        return names, values


class Experiment8Trial(Trial):
    def _choose_test_result(self):
        return Experiment8Result

    def _change_model(self):
        self.model.normalize()

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        method = self.setup.parameters["method"]
        uq = fitted_model.uq(method=method, options={"nsamples": 100})
        return uq


class Experiment8(Experiment):

    def _set_child_test(self):
        return Experiment8Trial

    def _setup_tests(self):
        setup_list = []
        uq_type_list = ["mc", "fci"]
        for uq_type in uq_type_list:
            setup = TestSetup(parameters={"method": uq_type})
            setup_list.append(setup)
        return setup_list