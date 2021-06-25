"""
Experiment 2: Test how close the initial guess for theta_v must be to the truth so that we still achieve good
reconstructions. We test this by adding random noise of different sizes to the true parameter theta_v_true,
and then fit the nonlinear model with the noisy value as initial guess (prior mean).
"""

from experiment_kit import SuperTest, TestResult, TestSetup
from nonlinear_test import NonlinearTest
from uq4pk_fit.model_fit import *


class ThetaGuessResult(TestResult):
    def __init__(self, q, niter, map_cost, truth_cost, rmisfit, recerr_f, recerr_theta_v):
        TestResult.__init__(self)
        self.names = ["q", "niter", "MAP-cost", "truth-cost", "rdm", "erel f",
                      "erel theta"]
        self.attributes = [q, niter, map_cost, truth_cost, rmisfit, recerr_f, recerr_theta_v]


class ThetaGuessTest(NonlinearTest):
    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        q = self.setup["theta_noise"]
        niter = fitted_model.info["niter"]
        f_map = fitted_model.f_map
        theta_v_map = fitted_model.theta_v_map
        costfun = fitted_model.rare_costfun
        map_cost = costfun(f_map.flatten(), theta_v_map)
        truth_cost = costfun(self.f.flatten(), self.theta_v)
        rmisfit = self._rdm(f_map, theta_v_map)
        rerr_f = self._err_f(f_map)
        rerr_theta_v = self._err_theta_v(theta_v_map)
        result = ThetaGuessResult(q=q, niter=niter, map_cost=map_cost, truth_cost=truth_cost, rmisfit=rmisfit,
                                  recerr_f=rerr_f, recerr_theta_v=rerr_theta_v)
        return result

    def _change_model(self):
        self.model.solveroptions["maxiter"] = 200

    def _quantify_uncertainty(self, fitted_model: FittedPixelModel):
        # turned off
        pass


class ThetaGuessSuperTest(SuperTest):

    def __init__(self, output_directory, f_list):
        SuperTest.__init__(self, output_directory, f_list)
        self.ChildTest = ThetaGuessTest

    def _setup_tests(self):
        setup_list = []
        q_list = [0., 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
        snr = 200
        for q in q_list:
            name = f"q={q}"
            setup = TestSetup(name=name, parameters={"theta_noise": q, "snr": snr})
            setup_list.append(setup)
        return setup_list


# ------------------------------------------------------------------- RUN
name = "experiment2"
list_of_f = get_f("data")
super_test = ThetaGuessSuperTest(output_directory=name, f_list=list_of_f)
super_test.perform_tests()
