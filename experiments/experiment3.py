"""
Experiment 3: In this experiment, we test how sensitive the reconstruction is to the regularization parameter alpha2
for theta_v. To this end, we fit the nonlinear model with an (again noisy) initial guess for theta_v but different
values for the regularization parameter alpha2.
"""

from uq4pk_fit.model_fit import *

from experiment_kit import SuperTest, TestResult, TestSetup
from nonlinear_test import NonlinearTest


class BetaTestResult(TestResult):
    def __init__(self, beta2, niter, map_cost, truth_cost, relative_misfit, f_recerr, theta_v_recerr):
        TestResult.__init__(self)
        self.names = ["beta2", "Number of iterations", "Cost at MAP", "Cost at truth", "Relative data misfit",
                      "Relative reconstruction error for f", "Relative reconstruction error for theta_v"]
        self.attributes = [beta2, niter, map_cost, truth_cost, relative_misfit, f_recerr, theta_v_recerr]


class BetaTest(NonlinearTest):

    def _change_model(self):
        self.model.alpha2 = self.setup["beta2"]
        self.model.solveroptions["maxiter"] = 10

    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        beta2 = self.setup["beta2"]
        niter = fitted_model.info["niter"]
        f_map = fitted_model.f_map
        theta_v_map = fitted_model.theta_v_map
        costfun = fitted_model.rare_costfun
        map_cost = costfun(f_map.flatten(), theta_v_map)
        truth_cost = costfun(self.f.flatten(), self.theta_v)
        rmisfit = self._rdm(f_map, theta_v_map)
        rerr_f = self._err_f(f_map)
        rerr_theta_v = self._err_theta_v(theta_v_map)
        result = BetaTestResult(beta2=beta2, niter= niter, map_cost=map_cost, truth_cost=truth_cost,
                                relative_misfit=rmisfit, f_recerr=rerr_f, theta_v_recerr=rerr_theta_v)
        return result

    def _quantify_uncertainty(self, fitted_model: FittedPixelModel):
        # turned off
        pass


class BetaSuperTest(SuperTest):

    def __init__(self, output_directory, f_list):
        SuperTest.__init__(self, output_directory, f_list)
        self.ChildTest = BetaTest

    def _setup_tests(self):
        setup_list = []
        #beta2_list = [0.01, 0.1, 0.3, 1., 3., 10, 100, 1000, 10000, 100000]
        beta2_list = [0.01, 0.1, 1, 10, 100]
        snr = 100
        for beta2 in beta2_list:
            name = f"beta2={beta2}"
            setup = TestSetup(name=name, parameters={"beta2": beta2, "snr": snr})
            setup_list.append(setup)
        return setup_list


# ------------------------------------------------------------------- RUN
name = "experiment3"
list_of_f = get_f("data", maxno=1)
super_test = BetaSuperTest(output_directory=name, f_list=list_of_f)
super_test.perform_tests()
