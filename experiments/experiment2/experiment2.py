"""
Experiment 2: Test how close the initial guess for theta_v must be to the truth so that we still achieve good
reconstructions. We test this by adding random noise of different sizes to the true parameter theta_v_true,
and then fit the nonlinear inference with the noisy value as initial guess (prior mean).
"""

from uq4pk_fit.inference import *

from experiments.experiment_kit import *



class Experiment2Result(TrialResult):

    def _compute_results(self):
        names = ["mapcost", "truthcost", "rdm", "errorf",
                      "erortheta"]
        map_cost = self.cost_map
        truth_cost = self.cost_truth
        rdm = self.rdm
        err_f = self.err_f
        err_theta_v = self.sre_tv
        attributes = [map_cost, truth_cost, rdm, err_f, err_theta_v]
        return names, attributes

    def _additional_plotting(self, savename):
        pass


class Experiment2Trial(Trial):
    def _choose_test_result(self):
        return Experiment2Result

    def _change_model(self):
        self.model.beta1 = 100 * 1e4
        self.model.beta2 = 1

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        # turned off
        pass


class Experiment2(Experiment):

    def _set_child_test(self):
        return Experiment2Trial

    def _setup_tests(self):
        setup_list = []
        setup = TestSetup({})
        setup_list.append(setup)
        return setup_list