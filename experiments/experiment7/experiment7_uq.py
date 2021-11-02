"""
Experiment 6: This experiment compares the reconstruction error for three different cases:
    1) Fixing theta_v to the true value, which means that the resulting inference is linear.
    2) Fixing theta_v partially, namely only h_0, h_1, h_2.
    3) Not fixing theta_v at all, but working with a reasonably good initial guess.
"""
import numpy as np

from uq4pk_fit.inference import *
from experiments.experiment_kit import *
from experiments.experiment7.experiment7 import Experiment7Trial, Experiment7


class Experiment7UQResult(TrialResult):
    def _compute_results(self):
        names = ["uqerrorf", "uqtightnessf", "uqerrortheta", "uqtightnesstheta"]
        # evaluate errors
        uq_error_f = self.uqerr_f
        uq_tightness_f = self.uqtightness_f
        uq_error_theta = self.uqerr_theta
        uq_tightness_theta = self.uqtightness_theta
        attributes = [uq_error_f, uq_tightness_f, uq_error_theta, uq_tightness_theta]
        return names, attributes

    def _additional_plotting(self, savename):
        pass
        # plot kernel functionals of ground truth
        f_truth = self._f_true
        f_map = self._fitted_model.f_map
        filter = self._uq.filter_f
        phi_true = filter.enlarge(filter.evaluate(f_truth))
        phi_map = filter.enlarge(filter.evaluate(f_map))
        phi_true_image = self._image(phi_true)
        phi_map_image = self._image(phi_map)
        vmax = np.max(f_truth)
        plot_with_colorbar(image=phi_true_image, vmax=vmax, savename=f"{savename}/filtered_truth.png")
        plot_with_colorbar(image=phi_map_image, vmax=vmax, savename=f"{savename}/filtered_map.png")


class Experiment7UQTest(Experiment7Trial):
    def _choose_test_result(self):
        return Experiment7UQResult

    # only thing that changes is the uncertainty quantification
    def _quantify_uncertainty(self, fitted_model: FittedModel):
        uq = fitted_model.uq(method="fci")
        return uq


class Experiment7UQ(Experiment7):

    def _set_child_test(self):
        return Experiment7UQTest

    def _setup_tests(self):
        setup_list = []
        fixation_list = ["fixed", "partially fixed", "not fixed"]
        for fixation in fixation_list:
            setup = TestSetup(parameters={"fixation": fixation})
            setup_list.append(setup)
        return setup_list
