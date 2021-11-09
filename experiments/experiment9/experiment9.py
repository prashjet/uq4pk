"""
This experiment reconstruct an age-metallicity distribution from real data.
The results are then compared to a baseline computed with PPXF.
The uncertainty quantification is performed both with Pereyra and RML.
"""

import numpy as np

from uq4pk_fit.inference import *
import uq4pk_fit.special_operators as so

from experiment_kit import *


class Test9(Test):

    def _read_setup(self, setup: dict):
        pass

    def _create_name(self) -> str:
        test_name = "test9"
        return test_name

    def _change_model(self):
        self.model.beta1 = 100
        self.model.beta2 = 1
        h = np.array([1, 1])
        self.model.P1 = so.OrnsteinUhlenbeck(m=self.model.m_f, n=self.model.n_f, h=h)
        # let's fix theta partially
        fixed_indices = np.array([2, 3, 4]) # corresponding to h_0, h_1, h_2
        fixed_values = np.array([1., 0., 0.])
        self.model.fix_theta_v(indices=fixed_indices, values=fixed_values)

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        pass


class Supertest9(SuperTest):

    _ChildTest = Test9

    def _setup_tests(self):
        setup_list = []
        setup = {}
        setup_list.append(setup)
        return setup_list