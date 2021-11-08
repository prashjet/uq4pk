"""
Experiment 6: This experiment compares the reconstruction error for three different cases:
    1) Fixing theta_v to the true value, which means that the resulting inference is linear.
    2) Fixing theta_v partially, namely only h_0, h_1, h_2.
    3) Not fixing theta_v at all, but working with a reasonably good initial guess.
"""

import enum
import numpy as np

from uq4pk_fit.inference import *

from experiments.experiment_kit import *


class Fixation(enum.Enum):
    fixed = "fixed"
    partially_fixed = "partially fixed"
    not_fixed = "not fixed"


class Experiment7Result(TrialResult):
    def _compute_results(self):
        names = ["errorf", "ssimf", "errortheta"]
        err_f = self.err_f
        ssimf = self.ssim_f
        sreless = self.sre_tv_less
        attributes = [err_f, ssimf, sreless]
        return names, attributes


class Experiment7Trial(Trial):
    def _choose_test_result(self):
        return Experiment7Result

    def _change_model(self):
        self.model.normalize()
        self.model.beta2 = 10   # just for protocol
        fixation = self.setup.parameters["fixation"]
        if fixation == Fixation.fixed.value:
            # fix theta_v to the ground truth
            fixed_indices = np.arange(7)
            self.model.fix_theta_v(fixed_indices, self.theta_guess)
        elif fixation == Fixation.partially_fixed.value:
            # set only h_0, h_1, h_2 (corresponding to theta_v[2:4]) equal to the true values
            fixed_indices = np.array([2, 3, 4])
            fixed_values = self.theta_guess[fixed_indices]
            self.model.fix_theta_v(fixed_indices, fixed_values)
            # if theta_v is not fixed, then one has to regularized f more
        elif fixation == Fixation.not_fixed.value:
            # do not fix anything
            pass
        else:
            raise KeyError("Unknown type.")

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Experiment7(Experiment):

    def _set_child_test(self):
        return Experiment7Trial

    def _setup_tests(self):
        setup_list = []
        fixation_list = ["fixed", "partially fixed", "not fixed"]
        for fixation in fixation_list:
            setup = TestSetup(parameters={"fixation": fixation})
            setup_list.append(setup)
        return setup_list