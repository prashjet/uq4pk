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

from uq4pk_fit.model_fit import *
from experiment_kit import TestResult, SuperTest, TestSetup
from linear_test import LinearTest

from uq4pk_fit.regop import *


class RegopTestResult(TestResult):
    def __init__(self, opname, niter, map_cost, truth_cost, relative_misfit, f_recerr):
        TestResult.__init__(self)
        self.names = ["regop", "niter", "MAP-cost", "Truth-cost",
                      "rdm", "erel f"]
        self.attributes = [opname, niter, map_cost, truth_cost, relative_misfit, f_recerr]


class RegopTest(LinearTest):

    def _change_model(self):
        regop = self.setup["regop"]
        if regop == "Identity":
            self.model.regop1 = TrivialOperator(dim=self.model.dim_f)
        elif regop == "Ornstein-Uhlenbeck":
            h = np.array([0.3, 2])
            self.regop1 = OrnsteinUhlenbeck(n1=self.model.dim_f1, n2=self.model.dim_f2, h=h)
        elif regop == "Discrete-Gradient":
            self.model.regop1 = DiscreteGradient(n1=self.model.dim_f1, n2=self.model.dim_f2)
        elif regop == "Discrete-Laplacian":
            self.model.regop1 = DiscreteLaplacian(n1=self.model.dim_f1, n2=self.model.dim_f2)
        self.model.solveroptions["maxiter"] = 150

    def _quantify_uncertainty(self, fitted_model: FittedLinearModel):
        # want no uncertainty quantification
        pass

    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        opname = self.setup["regop"]
        niter = fitted_model.info["niter"]
        f_map = fitted_model.f_map
        costfun = fitted_model.rare_costfun
        map_cost = costfun(f_map)
        truth_cost = costfun(self.f)
        rmisfit = self._rdm(f_map)
        rerr_f = self._err_f(f_map)
        result = RegopTestResult(opname=opname, niter=niter, map_cost=map_cost, truth_cost=truth_cost,
                                 relative_misfit=rmisfit, f_recerr=rerr_f)
        return result


class RegopSuperTest(SuperTest):

    def __init__(self, output_directory, f_list):
        SuperTest.__init__(self, output_directory, f_list)
        self.ChildTest = RegopTest

    def _setup_tests(self):
        setup_list = []
        regop_list = ["Identity", "Ornstein-Uhlenbeck", "Discrete-Gradient", "Discrete-Laplacian"]
        snr_list = [2000, 100]
        for snr in snr_list:
            for regop in regop_list:
                testname = f"snr={snr}_{regop}"
                setup = TestSetup(name=testname, parameters={"regop": regop, "snr": snr})
                setup_list.append(setup)
        return setup_list


# ------------------------------------------------------------------- RUN
name = "experiment1"
list_of_f = get_f("data")
super_test = RegopSuperTest(output_directory=name, f_list=list_of_f)
super_test.perform_tests()
