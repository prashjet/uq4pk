"""
Tests different type of uncertainty quantification for the Astro-problem.
"""

import numpy as np

from uq4pk_fit.inference import *

from experiments.experiment_kit import *


class Experiment6Result(TrialResult):
    def _compute_results(self):
        names = ["uqerror", "uqtightness"]
        attributes = [self.uqerr_f, self.uqtightness_f]
        return names, attributes

    def _additional_plotting(self, savename):
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


class Experiment6Trial(Trial):

    def _choose_test_result(self):
        return Experiment6Result

    def _change_model(self):
        self.model.fix_theta_v(self.theta_true)

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        kernel = self.setup.parameters["kernel"]
        uq = fitted_model.uq(method="fci", options={"kernel": kernel})
        return uq


class Experiment6(Experiment):

    def _set_child_test(self):
        return Experiment6Trial

    def _setup_tests(self):
        setup_list = []
        kernel_list = ["exp", "sqexp"]
        for kernel in kernel_list:
            setup = TestSetup(parameters={"kernel": kernel})
            setup_list.append(setup)
        return setup_list