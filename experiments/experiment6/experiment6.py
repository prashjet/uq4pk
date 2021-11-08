"""
Tests the effect of normalization.
"""

import numpy as np

from uq4pk_fit.inference import *

from experiments.experiment_kit import *


class Experiment6Result(TrialResult):
    def _compute_results(self):
        names = ["errof", "ssimf", "errortheta", "uqerrorf", "uqtightnessf", "uqerrortheta", "uqtightnesstheta"]
        err_f = self.err_f
        ssim_f = self.ssim_f
        err_theta = self.sre_tv
        uq_err_f = self.uqerr_f
        uq_tightness_f = self.uqtightness_f
        uq_error_theta = self.uqerr_theta
        uq_tightness_theta = self.uqtightness_theta
        attributes =  [err_f, ssim_f, err_theta, uq_err_f, uq_tightness_f, uq_error_theta, uq_tightness_theta]
        return names, attributes


class Experiment6Trial(Trial):

    def _choose_test_result(self):
        return Experiment6Result

    def _change_model(self):
        normalize = self.setup.parameters["normalize"]
        if normalize:
            self.model.normalize()

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci", options={"kernel": "laplace"})
        return uq


class Experiment6(Experiment):

    def _set_child_test(self):
        return Experiment6Trial

    def _setup_tests(self):
        setup_list = []
        on_off = [True, False]
        for normalize in on_off:
            setup = TestSetup(parameters={"normalize": normalize})
            setup_list.append(setup)
        return setup_list