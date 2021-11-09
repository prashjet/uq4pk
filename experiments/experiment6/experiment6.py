"""
Tests the effect of normalization.
"""

from uq4pk_fit.inference import *

from experiment_kit import *


class Test6(Test):

    def _read_setup(self, setup: dict):
        self.normalize = setup["normalize"]

    def _create_name(self) -> str:
        if self.normalize:
            test_name = "normalized"
        else:
            test_name = "unnormalized"
        return test_name

    def _change_model(self):
        if self.normalize:
            self.model.normalize()

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci", options={"kernel": "laplace"})
        return uq


class Supertest6(SuperTest):

    _ChildTest = Test6

    def _setup_tests(self):
        setup_list = []
        on_off = [True, False]
        for normalize in on_off:
            setup = {"normalize": normalize}
            setup_list.append(setup)
        return setup_list