"""
This experiment compares the filtered credible intervals computed via the Pereyra approximation to the ones
computed via RML (as a baseline).
The basic model is the nonlinear one, with theta_v not fixed at all.
"""

from uq4pk_fit.inference import *
from experiment_kit import *


class Test8(Test):

    def _read_setup(self, setup: dict):
        self.method = setup["method"]

    def _create_name(self) -> str:
        test_name = str(self.method)
        return test_name

    def _change_model(self):
        self.model.normalize()

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        method = self.method
        uq = fitted_model.uq(method=method, options={"nsamples": 10})
        return uq


class Supertest8(SuperTest):

    _ChildTest = Test8

    def _setup_tests(self):
        setup_list = []
        #uq_type_list = ["mc", "fci"]
        uq_type_list = ["mc"]
        for uq_type in uq_type_list:
            setup = {"method": uq_type}
            setup_list.append(setup)
        return setup_list