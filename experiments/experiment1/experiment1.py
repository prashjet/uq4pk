"""
Experiment 1: We test different regularization operators against each other for the reconstruction
of the age-metallicity distribution.
Since we are only interested in the regularization for the distribution function f, we assume that theta_v is known.
We test 4 different regularization operators:
- the identity matrix;
- the operator associated to an Ornstein-Uhlenbeck-type covariance matrix (the regularization operator is the
  square-root of the inverse);
- the discrete gradient;
- the discrete Laplacian;
"""

import numpy as np

from uq4pk_fit.cgn import IdentityOperator
from uq4pk_fit.special_operators import DiscreteLaplacian, DiscreteGradient, OrnsteinUhlenbeck
from uq4pk_fit.inference import *
from experiments.experiment_kit import *


class Experiment1Result(TrialResult):
    def _compute_results(self):
        names = ["mapcost", "truthcost",
                 "rdm", "errorf", "ssim"]
        map_cost = self.cost_map
        truth_cost = self.cost_truth
        rdm = self.rdm
        err_f = self.err_f
        ssim = self.ssim_f
        attributes = [map_cost, truth_cost, rdm, err_f, ssim]
        return names, attributes

    def _additional_plotting(self, savename):
        pass


class Experiment1Trial(Trial):

    def _choose_test_result(self):
        return Experiment1Result

    def _change_model(self):
        # set inference to linear
        self.model.fix_theta_v(indices=np.arange(self.dim_theta), values=self.theta_true)
        snr = self.model.snr
        regop = self.setup.parameters["regop"]
        if regop == "Identity":
            self.model.P1 = IdentityOperator(dim=self.model.dim_f)
            self.model.beta1 = snr * 1e3
        elif regop == "OrnsteinUhlenbeck":
            h = np.array([4, 2])
            self.model.P1 = OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
            self.model.beta1 = snr * 1e3
        elif regop == "DiscreteGradient":
            self.model.P1 = DiscreteGradient(m=self.model.m_f, n=self.model.n_f)
            self.model.beta1 = snr * 1e3
        elif regop == "DiscreteLaplacian":
            self.model.P1 = DiscreteLaplacian(m=self.model.m_f, n=self.model.n_f)
            self.model.beta1 = snr * 1e3

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # want no uncertainty quantification
        pass


class Experiment1(Experiment):

    def _set_child_test(self):
        return Experiment1Trial

    def _setup_tests(self):
        setup_list = []
        regop_list = ["Identity", "OrnsteinUhlenbeck", "DiscreteGradient", "DiscreteLaplacian"]
        for regop in regop_list:
            setup = TestSetup({"regop": regop})
            setup_list.append(setup)
        return setup_list
