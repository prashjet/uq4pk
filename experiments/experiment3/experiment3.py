"""
Experiment 3: In this experiment, we test how sensitive the reconstruction is to the regularization parameter beta1.
To this end, we fit the nonlinear inference with an (again noisy) initial guess for theta_v but different
values for the regularization parameter beta1.
"""

from uq4pk_fit.inference import *

from experiment_kit import *

class Test3(Test):

    def _read_setup(self, setup: dict):
        self.beta1 = setup["beta1"]

    def _create_name(self) -> str:
        test_name = f"{self.beta1}"
        return test_name

    def _change_model(self):
        self.model.beta1 = self.beta1

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Supertest3(SuperTest):

    _ChildTest = Test3

    def _setup_tests(self):
        setup_list = []
        beta1_list = [100, 500, 1000, 2000]
        for beta1 in beta1_list:
            setup = {"beta1": beta1 * 1e2}
            setup_list.append(setup)
        return setup_list
