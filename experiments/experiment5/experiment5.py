"""
Experiment 5: A second test for the uncertainty quantification, but this time for the nonlinear inference. This means
we also obtain local credible intervals for theta_v, which we evaluate graphically and numerically.
"""

from uq4pk_fit.inference import *

from experiment_kit import *


class Test5(Test):

    def _read_setup(self, setup: dict):
        pass

    def _create_name(self) -> str:
        test_name = f"nonlinear"
        return test_name

    def _change_model(self):
        pass

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci")
        return uq


class Supertest5(SuperTest):

    def _set_child_test(self):
        return Supertest5

    def _setup_tests(self):
        setup_list = []
        setup = {}
        setup_list.append(setup)
        return setup_list
