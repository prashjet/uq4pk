"""
This is a tutorial for the usage of ``experiment_kit``.
"""

from typing import List

import experiment_kit as ek
from uq4pk_fit.inference import FittedModel, UQResult


# First, make a test by inheriting from experiment_kit.Test
class DemoTest(ek.Test):

    # First, we specify the parameters of the test by implementing the "_read_setup" function.
    def _read_setup(self, setup: dict):
        self.param1 = setup["param1"]
        self.param2 = setup["param2"]

    # Next, we define a name for the test by implementing the function "_create_name". Note that this function can use
    # the parameters that are initialized in "_read_kwargs".
    def _create_name(self) -> str:
        test_name = f"test_with_param1={self.param1}"
        return test_name

    # Next, we have to implement the model by making some changes. These changes may use the parameters defined in
    # the "read_kwargs" function.
    # If you only want to use the default model, implement it as follows:
    #   _def change_model(self)
    #       pass
    def _change_model(self):
        # The model is referenced by self.model
        self.model.beta1 = self.param1
        if self.param2:
            self.model.normalize()

    # Similarly, you can control how to uncertainty quantification is performed by implementing the function
    # "_quantify_uncertainty". This function must return an object of type uq4pk_fit.inference.UQResult
    # If you do not want to perform uncertainty quantification, you can just pass.
    def _quantify_uncertainty(self, fitted_model: FittedModel) -> UQResult:
        uq = fitted_model.uq(method="fci", options={"kernel": "gauss"})
        return uq


# Next, we make a supertest by inheriting from SuperTest.
# A supertest manages a whole list of tests with different parameter values.
class DemoSupertest(ek.Supertest):

    # First, we define what kind of tests the supertest manages by setting the "ChildTest" parameter
    _ChildTest = DemoTest

    # Then, we define the list of tests by implementing "_setup_tests", which has to return a list.
    # Each entry of the list is a dict corresponding to an argument for Test1.
    def _setup_tests(self) -> List[dict]:
        testsetup_list = []
        param1_values = [0, 1, 2]
        param2_values = [False, True]
        for param1 in param1_values:
            for param2 in param2_values:
                setup = {"param1": param1, "param2": param2}
                testsetup_list.append(setup)
        return testsetup_list
