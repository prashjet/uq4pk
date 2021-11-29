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
from typing import List

from uq4pk_fit.cgn import IdentityOperator
from uq4pk_fit.special_operators import DiscreteLaplacian, DiscreteGradient, OrnsteinUhlenbeck
from uq4pk_fit.inference import *
from experiment_kit import *


class Test1(Test):

    def _read_setup(self, setup: dict):
        self.regop = setup["regop"]

    def _create_name(self) -> str:
        test_name = f"{self.regop}"
        return test_name

    def _change_model(self):
        # set inference to linear
        self.model.fix_theta_v(indices=np.arange(self.dim_theta), values=self.theta_true)
        snr = self.model.snr
        regop = self.regop
        if regop == "Identity":
            self.model.P1 = IdentityOperator(dim=self.model.dim_f)
            self.model.beta1 = snr * 1e3
        elif regop == "OrnsteinUhlenbeck":
            h = np.array([4, 2])
            self.model.P1 = OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
            self.model.beta1 = snr * 1e3
        elif regop == "DiscreteGradient":
            self.model.P1 = DiscreteGradient()
            self.model.beta1 = snr * 1e3
        elif regop == "DiscreteLaplacian":
            self.model.P1 = DiscreteLaplacian(m=self.model.m_f, n=self.model.n_f)
            self.model.beta1 = snr * 1e3

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # want no uncertainty quantification
        pass


class SuperTest1(SuperTest):

    _ChildTest = Test1

    def _setup_tests(self) -> List[dict]:
        setup_list = []
        #regop_list = ["Identity", "OrnsteinUhlenbeck", "DiscreteGradient", "DiscreteLaplacian"]
        regop_list = ["OrnsteinUhlenbeck", "DiscreteGradient"]
        for regop in regop_list:
            setup = {"regop": regop}
            setup_list.append(setup)
        return setup_list
