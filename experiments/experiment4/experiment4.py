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
from experiment_kit import *


class Test4(Test):

    def _read_setup(self, setup: dict):
        self.regop = setup["regop"]

    def _create_name(self) -> str:
        test_name = f"{self.regop}"
        return test_name

    def _change_model(self):
        # want linear inference
        self.model.fix_theta_v(indices=np.arange(self.dim_theta), values=self.theta_true)
        regop = self.regop
        if regop == "Identity":
            self.model.P1 = IdentityOperator(dim=self.model.dim_f)
            self.model.beta1 = 1e4
        elif regop == "OrnsteinUhlenbeck":
            h = np.array([6, 2])
            self.model.normalize()
            self.model.P1 = OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
            #self.model.beta1 = 100 * 1e4
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


class Supertest4(SuperTest):

    _ChildTest = Test4

    def _setup_tests(self):
        setup_list = []
        #regop_list = ["Identity", "OrnsteinUhlenbeck", "Gradient", "Laplacian"]
        regop_list = ["OrnsteinUhlenbeck"]
        for regop in regop_list:
            setup = {"regop": regop}
            setup_list.append(setup)
        return setup_list
