"""
Experiment 3: In this experiment, we test how sensitive the reconstruction is to the regularization parameter beta1.
To this end, we fit the nonlinear inference with an (again noisy) initial guess for theta_v but different
values for the regularization parameter beta1.
"""

from uq4pk_fit.inference import *

from experiment_kit import *

class Test3(Test):

    def _read_setup(self, setup: dict):
        self.beta2 = setup["beta2"]

    def _create_name(self) -> str:
        test_name = f"{self.beta2}"
        return test_name

    def _change_model(self):
        self.model.beta2 = self.beta2

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Supertest3(SuperTest):

    _ChildTest = Test3

    def _setup_tests(self):
        setup_list = []
        beta2_list = [0.1, 1., 10., 100, 1000]
        for beta2 in beta2_list:
            setup = {"beta2": beta2}
            setup_list.append(setup)
        return setup_list
