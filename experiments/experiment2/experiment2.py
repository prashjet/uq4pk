"""
Experiment 2: Test how close the initial guess for theta_v must be to the truth so that we still achieve good
reconstructions. We test this by adding random noise of different sizes to the true parameter theta_v_true,
and then fit the nonlinear inference with the noisy value as initial guess (prior mean).
"""

from experiment_kit import *
from uq4pk_fit.inference import FittedModel, UQResult


class Test2(Test):

    def _read_setup(self, setup: dict):
        pass

    def _create_name(self) -> str:
        test_name = "guess_test"
        return test_name

    def _change_model(self):
        pass

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Supertest2(SuperTest):

    _ChildTest = Test2

    def _setup_tests(self):
        only_setup = {}
        return [only_setup]