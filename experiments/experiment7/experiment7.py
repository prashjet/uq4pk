"""
Experiment 6: This experiment compares the reconstruction error for three different cases:
    1) Fixing theta_v to the true value, which means that the resulting inference is linear.
    2) Fixing theta_v partially, namely only h_0, h_1, h_2.
    3) Not fixing theta_v at all, but working with a reasonably good initial guess.
"""

import numpy as np

from uq4pk_fit.inference import *

from experiment_kit import *


class Test7(Test):

    def _read_setup(self, setup: dict):
        self.fixation = setup["fixation"]

    def _create_name(self) -> str:
        test_name = str(self.fixation)
        return test_name

    def _change_model(self):
        self.model.normalize()
        fixation = self.fixation
        if fixation == "fixed":
            # fix theta_v to the ground truth
            fixed_indices = np.arange(7)
            self.model.fix_theta_v(fixed_indices, self.theta_guess)
        elif fixation == "partially fixed":
            # set only h_0, h_1, h_2 (corresponding to theta_v[2:4]) equal to the true values
            fixed_indices = np.array([2, 3, 4])
            fixed_values = self.theta_guess[fixed_indices]
            self.model.fix_theta_v(fixed_indices, fixed_values)
            # if theta_v is not fixed, then one has to regularized f more
        elif fixation == "not fixed":
            # do not fix anything
            pass
        else:
            raise KeyError("Unknown type.")

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Supertest7(SuperTest):

    _ChildTest = Test7

    def _setup_tests(self):
        setup_list = []
        fixation_list = ["fixed", "partially fixed", "not fixed"]
        for fixation in fixation_list:
            setup = {"fixation": fixation}
            setup_list.append(setup)
        return setup_list