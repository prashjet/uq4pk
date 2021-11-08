"""
Contains class "Trial"
"""

import numpy as np
import os

from uq4pk_fit.inference import *
from uq4pk_fit.cgn import DiagonalOperator
from .experiment_data import ExperimentData
from .trial_result import TrialResult
from .test_setup import TestSetup


class Trial:
    """
    ABSTRACT BASE CLASS FOR TESTS.
    For a concrete test, one only has to implement _change_model, _make_testresult, _quantify_uncertainty
    """
    def __init__(self, outname, data: ExperimentData, setup: TestSetup):
        self.outname = outname
        self.setup = setup
        # READ DATA
        self.f = data.f_true
        self.f_ref = data.f_ref
        self.theta_true = data.theta_true
        self.theta_guess = data.theta_guess
        self.theta_sd = data.theta_sd
        self.y = data.y
        self.y_sd = data.y_sd
        # compute snr
        self.snr = np.linalg.norm(self.y) / np.linalg.norm(self.y_sd)
        self.op = data.forward_operator
        self.grid = self.op.grid
        # INITIALIZE STATMODEL
        self.model = self._initialize_model()
        self.dim_f = self.model.dim_f
        self.dim_theta = self.model.dim_theta
        # define ResultType for child.
        self.ResultType = self._choose_test_result()

    # ABSTRACT METHODS --- IMPLEMENT!

    def _choose_test_result(self):
        raise NotImplementedError

    def _change_model(self):
        """
        Here, you can adapt the basic inference. For example, you can add constraints, change the regularization, fix
        theta_v (partially or completely) and define solveroptions.
        """
        raise NotImplementedError

    def _quantify_uncertainty(self, fitted_model: FittedModel) -> UQResult:
        """
        Here, the user can define how and if the uncertainty is quantified.
        If you want to turn uncertainty quantification off, just return None.
        """
        raise NotImplementedError

    # CONCRETE METHODS --- DO NOT OVERWRITE!

    def _make_testresult(self, fitted_model: FittedModel, uq) -> TrialResult:
        test_result = self.ResultType(savename=self.outname, setup=self.setup,
                                      fitted_model=fitted_model,
                                      statmodel=self.model, uq=uq, f_true=self.f, f_ref=self.f_ref,
                                      theta_true=self.theta_true, theta_ref=self.theta_guess)
        return test_result

    def _initialize_model(self) -> StatModel:
        """
        Initializes a concrete instance of "Model". The inference can then be changed afterwards by adapting the
        function "_change_model".
        :return:
        """
        y = self.y
        y_sd = self.y_sd
        op = self.op
        model = StatModel(y=y, y_sd=y_sd, forward_operator=op)
        model.theta_bar = self.theta_guess
        model.P2 = DiagonalOperator(dim=model.dim_theta, s=np.divide(1, self.theta_sd))
        return model

    def do_test(self) -> TrialResult:
        """
        Fits the inference using the solver, and computes the uncertainty quantification.
        :return: TrialResult
        """
        self._change_model()
        # self._show_model_info()
        fitted = self.model.fit()
        # Compute uncertainty quantification
        uq = self._quantify_uncertainty(fitted)
        if uq is None:
            uq = NullUQResult()
        # Make TestResult object and return it
        testresult = self._make_testresult(fitted, uq)
        return testresult
