"""
Experiment 5: A second test for the uncertainty quantification, but this time for the nonlinear model. This means
we also obtain local credible intervals for theta_v, which we evaluate graphically and numerically.
"""

import numpy as np

from uq4pk_fit.regop import *
from uq4pk_fit.uq_mode import *

from uq4pk_fit.model_fit import *
from experiment_kit import TestResult, SuperTest, TestSetup
from nonlinear_test import NonlinearTest
from plotting_kit import *


class UQNLTestResult(TestResult):
    def __init__(self, theta_noise, snr, regop, truth_credible, e_rci_f, e_rci_theta, rci_size_f, rci_size_theta):
        TestResult.__init__(self)
        self.names = ["theta_noise", "snr", "operator", "truth in CR", "LCI error f", "LCI error theta", "LCI size f",
                      "LCI size theta"]
        self.attributes = [theta_noise, snr, regop, truth_credible, e_rci_f, e_rci_theta, rci_size_f, rci_size_theta]


class UQNLTest(NonlinearTest):

    def __init__(self, outname, f, setup):
        NonlinearTest.__init__(self, outname, f, setup)

    def _change_model(self):
        self.model.solveroptions["maxiter"] = 10
        regop = self.setup["regop"]
        if regop == "Identity":
            self.model.regop1 = TrivialOperator(dim=self.model.dim_f)
        elif regop == "Ornstein-Uhlenbeck":
            h = np.array([0.3, 2])
            self.regop1 = OrnsteinUhlenbeck(n1=self.model.dim_f1, n2=self.model.dim_f2, h=h)
        elif regop == "Gradient":
            self.model.regop1 = DiscreteGradient(n1=self.model.dim_f1, n2=self.model.dim_f2)
        elif regop == "Laplacian":
            self.model.regop1 = DiscreteLaplacian(n1=self.model.dim_f1, n2=self.model.dim_f2)
        else:
            raise Exception("Regop not recognized.")

    def _make_testresult(self, fitted_model: FittedPixelModel, credible_intervals) -> TestResult:
        f_map = fitted_model.f_map
        theta_v_map = fitted_model.theta_v_map
        x_map = np.concatenate((f_map, theta_v_map))
        x_true = np.concatenate((self.f, self.theta_v))
        truth_credible = is_credible(x=x_true, alpha=0.5, xmap=x_map, costfun=fitted_model.costfun)
        # compute quantitative error measures for the uncertainty quantification
        err_lci_f, err_lci_theta = self._uq_error(credible_intervals)
        fdim = self.model.dim_f
        lcr_ratios_f = np.divide(np.abs(credible_intervals[:fdim,1]-credible_intervals[:fdim,0]),
                                 np.abs(self.f-f_map).clip(min=1))
        lcr_ratios_tv = np.divide(np.abs(credible_intervals[fdim:,1] - credible_intervals[fdim:,0]),
                                  np.abs(self.theta_v-theta_v_map).clip(min=0.1))
        s_uq_f = np.mean(lcr_ratios_f)
        s_uq_tv = np.mean(lcr_ratios_tv)
        result = UQNLTestResult(theta_noise=self.setup["theta_noise"],
                              snr=self.setup["snr"],
                              regop=self.setup["regop"],
                              truth_credible=truth_credible,
                              e_rci_f=err_lci_f,
                              e_rci_theta=err_lci_theta,
                              rci_size_f=s_uq_f,
                            rci_size_theta=s_uq_tv)
        return result

    def _uq_error(self, region):
        """
        Given a credible region [xmin, xmax], evaluates how much of the parameter xtrue=(f, theta_v)
        lies in [xmin, xmax].
        This is done by computing err_plus = (x - xmax)^+, err_minus = (x - xmin)^- and then
        err_uq = np.linalg.norm((err_plus; err_minus)).
        :return: float
            if 0, this means that x lies in [xmin, xmax]. Otherwise, indicates how much of x lies outside
            [xmin, xmax].
        """
        dim_f = self.model.dim_f
        fmin = region[:dim_f, 0]
        fmax = region[:dim_f, 1]
        thetamin = region[dim_f:, 0]
        thetamax = region[dim_f:, 1]
        ftrue = self.f
        thetatrue = self.theta_v
        def relative_outside(true, min, max):
            too_large = true - max
            too_small = true - min
            err_plus = self._positive_part(too_large)
            err_minus = self._positive_part(too_small)
            error = np.linalg.norm(np.concatenate((err_plus, err_minus)), ord=1) / np.linalg.norm(true, ord=1)
            return error
        err_f = relative_outside(ftrue, fmin, fmax)
        err_theta = relative_outside(thetatrue, thetamin, thetamax)
        return err_f, err_theta

    def _check_map_inside(self, xmap, xi):
        """
        Checks wheter xi[:,0] <= xmap <= xi[:,1]
        """
        if np.all(xi[:,0]<=xmap) and np.all(xmap<=xi[:,1]):
            print("MAP estimate is inside RCI")
        else:
            print("WARNING: MAP estimate outside RCI. This should not happen!!!")

    def _positive_part(self, x):
        """
        Returns the positive part of the vector x
        """
        return x.clip(min=0.)

    def _plot_uq(self, name, xi, thetamap, vmax=None):
        """
        Plots the uncertainty quantification
        """
        # visualize uncertainty quantification for f
        fmin = np.reshape(xi[:self.model._n_f, 0], (self.model._n_f_1, self.model._n_f_2))
        fmax = np.reshape(xi[:self.model._n_f, 1], (self.model._n_f_1, self.model._n_f_2))
        savename1 = f"{self.outname}/{name}_f_lower.png"
        savename2 = f"{self.outname}/{name}_f_upper.png"
        # determine vmax
        if vmax is None:
            vmax = np.max(xi.flatten())
        # plot the lower bound
        plot_with_colorbar(image=fmin, savename=savename1, vmax=vmax)
        plot_with_colorbar(image=fmax, savename=savename2, vmax=vmax)

        # visualize uncertainty quantification for theta_v
        thetamin = xi[self.model._n_f:, 1]
        thetamax = xi[self.model._n_f:, 0]
        # errorbars are the differences between x1 and x2
        below_error = thetamap-thetamin
        upper_error = thetamax-thetamap
        errorbars = np.row_stack((below_error, upper_error))
        savename5 = f"{self.outname}/{name}_V_and_sigma.png"
        savename6 = f"{self.outname}/{name}_h.png"
        names1 = ["V", "sigma"]
        plot_double_bar(safename=savename5, name_list=names1, values1=self.theta_v[:2],
                        values2=thetamap[:2], name1="Truth", name2="Estimate", errorbars=errorbars[:,:2])
        # plot second 4 variables
        names2 = ["h_0", "h_1", "h_2", "h_3", "h_4"]
        plot_double_bar(safename=savename6, name_list=names2, values1=self.theta_v[2:],
                        values2=thetamap[2:], name1="Truth", name2="Estimate", errorbars=errorbars[:,2:])




class UQNLSuperTest(SuperTest):

    def __init__(self, output_directory, f_list):
        SuperTest.__init__(self, output_directory, f_list)
        self.ChildTest = UQNLTest

    def _setup_tests(self):
        setup_list = []
        snr_list = [2000, 100]
        q = 0.03
        regop = "Ornstein-Uhlenbeck"
        for snr in snr_list:
            name = f"snr={snr}_{regop}"
            setup = TestSetup(name=name, parameters={"snr": snr, "theta_noise": q, "regop": regop})
            setup_list.append(setup)
        return setup_list


# ------------------------------------------------------------------- RUN
name = "experiment5"
list_of_f = get_f("data", maxno=1)
super_test = UQNLSuperTest(output_directory=name, f_list=list_of_f)
super_test.perform_tests()
