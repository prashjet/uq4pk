"""
Experiment 4: In this experiment we test the uncertainty quantification using local credible intervals based
on finding the smallest rectangle that contains the (localized) a-posteriori credible region.
We test the uncertainty quantification on the linear inference using different regularization operators
(see experiment 1).
The quality of the uncertainty quantification is evaluated graphically by producing visualizations, and numerically
by computing quantitative error measures that indicate whether our approach to uncertainty quantification accurately
represents the "true" posterior uncertainty.
"""

import numpy as np

from uq4pk_fit.inference import *
from uq4pk_fit.cgn import IdentityOperator
from uq4pk_fit.special_operators import DiscreteGradient, DiscreteLaplacian, OrnsteinUhlenbeck
from experiments.experiment_kit import *


class Experiment4Result(TrialResult):
    def _compute_results(self):
        names = ["uqerror", "uqtightness"]
        attributes = [self.uqerr_f, self.uqtightness_f]
        return names, attributes

    def _additional_plotting(self, savename):
        pass


class Experiment4Trial(Trial):

    def _choose_test_result(self):
        return Experiment4Result

    def _change_model(self):
        # want linear inference
        self.model.fix_theta_v(indices=np.arange(self.dim_theta), values=self.theta_true)
        regop = self.setup.parameters["regop"]
        if regop == "Identity":
            self.model.P1 = IdentityOperator(dim=self.model.dim_f)
            self.model.beta1 = 1e4
        elif regop == "OrnsteinUhlenbeck":
            h = np.array([6, 2])
            self.model.P1 = OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
            self.model.beta1 = 100 * 1e4
        elif regop == "Gradient":
            self.model.P1 = DiscreteGradient(m=self.model.m_f, n=self.model.n_f)
            self.model.beta1 = 1e4
        elif regop == "Laplacian":
            self.model.P1 = DiscreteLaplacian(m=self.model.m_f, n=self.model.n_f)
            self.model.beta1 = 1e4
        else:
            raise Exception("Regop not recognized.")

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci")
        return uq


class Experiment4(Experiment):

    def _set_child_test(self):
        return Experiment4Trial

    def _setup_tests(self):
        setup_list = []
        regop_list = ["Identity", "OrnsteinUhlenbeck", "Gradient", "Laplacian"]
        for regop in regop_list:
            setup = TestSetup({"regop": regop})
            setup_list.append(setup)
        return setup_list
