"""
This experiment reconstruct an age-metallicity distribution from real data.
The results are then compared to a baseline computed with PPXF.
The uncertainty quantification is performed both with Pereyra and RML.
"""

import numpy as np

from uq4pk_fit.inference import *
import uq4pk_fit.cgn as cgn

from experiments.experiment_kit import *


class Experiment9Result(TrialResult):
    def _compute_results(self):
        # no uq in phase 1
        names = ["rdm", "ferror", "fssim", "tverror"]
        # Evaluate reconstruction error
        f_error = self.err_f
        fssim = self.ssim_f
        tv_error = self.sre_tv
        attributes = [self.rdm, f_error, fssim, tv_error]
        return names, attributes

    def _additional_plotting(self, savename):
        pass


class Experiment9Trial(Trial):
    def _choose_test_result(self):
        return Experiment9Result

    def _change_model(self):
        self.model.beta1 = 1
        self.model.beta2 = 0.01
        h = np.array([1, 1])
        self.model.P1 = cgn.OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
        # let's fix theta partially
        fixed_indices = np.array([2, 3, 4]) # corresponding to h_0, h_1, h_2
        fixed_values = np.array([1., 0., 0.])
        self.model.fix_theta_v(indices=fixed_indices, values=fixed_values)


    def _quantify_uncertainty(self, fitted_model: FittedModel):
        pass
        #uq = self._compute_kci(fitted_model=fitted_model, kernel="exponential")
        #return uq



class Experiment9(Experiment):

    def _set_child_test(self):
        return Experiment9Trial

    def _setup_tests(self):
        setup_list = []
        setup = TestSetup({})
        setup_list.append(setup)
        return setup_list