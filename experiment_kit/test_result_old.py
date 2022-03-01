
from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
import skimage.metrics as scimet

from uq4pk_fit.cgn import IdentityOperator, MultipliedOperator, RegularizationOperator
from uq4pk_fit.inference import *
from uq4pk_fit.uq_mode import FilterFunction
from uq4pk_fit.plotting.plot_with_colorbar import plot_with_colorbar
from uq4pk_fit.plotting import plot_triple_bar
from .test_setup import TestSetup
from uq4pk_fit.inference.uq_result import UQResult


class TestResult:
    def __init__(self, savename, setup: TestSetup, fitted_model: FittedModel, statmodel: StatModel,
                 uq: UQResult,
                 f_true: ArrayLike, f_ref: ArrayLike, theta_true: ArrayLike, theta_ref: ArrayLike):
        # read input
        self._savename = savename
        self._setup = deepcopy(setup)
        self._fitted_model = deepcopy(fitted_model)
        self._statmodel = deepcopy(statmodel)
        self._f_true = f_true.copy()
        self._f_ref = f_ref.copy()
        self._theta_true = theta_true.copy()
        self._theta_ref = theta_ref.copy()
        self._uq = deepcopy(uq)

        # basic analysis
        self.rdm, self.cost_map, self.cost_truth, self.err_f, self.ssim_f, self.sre_tv, self.sre_tv_less \
            = self._error_analysis()

        # analysis of uncertainty quantification
        self.uqerr_f, self.uqtightness_f, self.uqsize_f, self.uqerr_theta, self.uqtightness_theta = self._uq_error_analysis()

        # default visualization
        self.plot()

        # additional visualization
        self._additional_plotting()

        # now, the actual results are computed
        self._names, self._values = self._compute_results()

    # ADAPT:

    def _compute_results(self):
        """
        :return: dict, dict
            Has to return two dicts of equal length. The first dict must contain a list of names of all the quantities
            of interest that constitute the test result., the second dict then contains a list of the corresponding
            values.
        """
        # here, you can compute all the things that you want to know
        raise NotImplementedError

    # ADAPT OPTIONALLY

    def _additional_plotting(self):
        scale = self._f_true.max()
        self._plot_f(name="scaled", scale=scale)

    # DO NOT ADAPT:

    def get_names(self):
        return self._names.copy()

    def get_values(self):
        """
        Returns all attributes as a list
        :return: list
        """
        return self._values.copy()

    def plot(self):
        self._plot_f()
        self._plot_theta_v()

    #   PROTECTED

    def _error_analysis(self):
        """
        """
        # Compute relative data misfit
        f_map = self._fitted_model.f_map
        theta_map = self._fitted_model.theta_map
        rdm = self._rdm(f_map, theta_map)
        rdm_ref = self._rdm(self._f_ref, self._theta_ref)     #
        rdm_truth = self._rdm(self._f_true, self._theta_true)  #
        # Compare the cost functions
        m = 1e4
        cost_ref = self._fitted_model.costfun(self._f_ref, self._theta_ref) / m
        cost_truth = self._fitted_model.costfun(self._f_true, self._theta_true) / m
        cost_map = self._fitted_model.costfun(f_map, theta_map) / m
        # Compute reconstruction error for f
        err_f, ssim_f = self._compute_error_for_f(f_map)
        err_ref, ssim_ref = self._compute_error_for_f(self._f_ref)
        # Compute reconstruction error for theta_v
        sre_theta, sre_theta_less = self._compute_error_for_theta()
        print(" ")
        print("ERROR ANALYSIS")
        print(f"Relative data misfit: {rdm}")
        print(f"Relative data misfit for reference {rdm_ref}")
        print(f"Relative data misfit for truth {rdm_truth}")
        print(f"Cost at reference parameter: {cost_ref}")
        print(f"Cost at true parameter: {cost_truth}")
        print(f"Cost at MAP estimate: {cost_map}")
        print(f"Relative reconstruction error for f: {err_f}")
        print(f"Relative reconstruction error of reference: {err_ref}")
        print(f"SSIM for f: {ssim_f}")
        print(f"SSIM for reference: {ssim_ref}")
        print(f"Scaled reconstruction error for theta: {sre_theta}")
        print(f"Scaled reconstruction error for vartheta: {sre_theta_less}")
        print(f"Sum of map: {np.sum(f_map)}")
        print(f"Sum of truth: {np.sum(self._f_true)}")
        return rdm, cost_map, cost_truth, err_f, ssim_f, sre_theta, sre_theta_less

    def _compute_error_for_f(self, f):
        """
        Performs complete error analysis for f, including uq if given.
        :return: float
            The relative reconstruction error of the normalized age-metallicity distribution.
        """
        f_true = self._f_true
        err_f = self._relative_error(map=f, truth=f_true)
        # Also compute SSIM
        f_true_im = self._image(f_true)
        f_im = self._image(f)
        try:
            ssim_f = scimet.structural_similarity(f_im, f_true_im,
                                                data_range=f_true_im.max() - f_true_im.min())
        except:
            ssim_f = -1
        return err_f, ssim_f

    def _compute_error_for_theta(self):
        """
        :return: float
            The relative reconstruction error for theta.
        """
        # Compute errors
        theta_map = self._fitted_model.theta_map
        P_theta = self._statmodel.P2
        sre_theta = self._scaled_error(map=theta_map, truth=self._theta_true, regop=P_theta)
        less = [0, 1, 5, 6]
        idmat = np.eye(theta_map.size)
        emb_mat = idmat[:, less]
        P_vartheta = MultipliedOperator(regop=P_theta, q=emb_mat)
        sre_theta_less = self._scaled_error(map=theta_map[less], truth=self._theta_true[less],
                                            regop=P_vartheta)
        return sre_theta, sre_theta_less

    def _plot_f(self, name = None, scale = None):
        if name is None:
            postfix = ""
        else:
            postfix = f"_{name}"
        # make images
        f_true_image = self._image(self._f_true)
        f_map_image = self._fitted_model.f_map
        f_ref_image = self._image(self._f_ref)
        f_all = np.concatenate((f_map_image.flatten(), f_ref_image.flatten(), f_true_image.flatten()))
        # determine v_max
        if scale is None:
            vmax = np.max(f_all)
        else:
            vmax = scale
        vmin = 0.
        # Plot true distribution function vs MAP estimate
        plot_with_colorbar(image=f_true_image, savename=f"{self._savename}/truth{postfix}", vmax=vmax, vmin=vmin)
        plot_with_colorbar(image=f_map_image, savename=f"{self._savename}/map{postfix}", vmax=vmax, vmin=vmin)
        plot_with_colorbar(image=f_ref_image, savename=f"{self._savename}/ref{postfix}", vmax=vmax, vmin=vmin)
        ci_f = self._uq.ci_f
        if ci_f is not None:
            f_min = ci_f[:, 0]
            f_max = ci_f[:, 1]
            f_min_image = self._image(f_min)
            f_max_image = self._image(f_max)
            if scale is None:
                vmax = f_max.max()
            else:
                vmax = scale
            f_size_image = self._image(f_max-f_min)
            plot_with_colorbar(image=f_min_image, savename=f"{self._savename}/lower{postfix}", vmax=vmax, vmin=vmin)
            plot_with_colorbar(image=f_max_image, savename=f"{self._savename}/upper{postfix}", vmax=vmax, vmin=vmin)
            plot_with_colorbar(image=f_size_image, savename=f"{self._savename}/size{postfix}", vmax=vmax, vmin=vmin)
            filter = self._uq.filter_f
            phi_true = filter.enlarge(filter.evaluate(self._f_true))
            phi_map = filter.enlarge(filter.evaluate(f_map))
            phi_true_image = self._image(phi_true)
            phi_map_image = self._image(phi_map)
            plot_with_colorbar(image=phi_true_image, vmax=vmax, savename=f"{self._savename}/filtered_truth{postfix}")
            plot_with_colorbar(image=phi_map_image, vmax=vmax, savename=f"{self._savename}/filtered_map{postfix}")
            # plot treshold map (1 = lower, 2 = upper)
            eps = vmax * 0.05
            lower_on = (f_min_image > eps).astype(int)
            upper_on = (f_max_image > eps).astype(int)
            treshold_image = lower_on + upper_on
            plot_with_colorbar(image=treshold_image, savename=f"{self._savename}/treshold")

    def _plot_theta_v(self):
        """
        Assuming that V and sigma are not fixed...
        """
        theta_map = self._fitted_model.theta_map
        theta_ref = self._theta_ref
        theta_true = self._theta_true
        ci_theta = self._uq.ci_theta
        if ci_theta is not None:
            # if we want uncertainty quantification, we compute the error bars from the local credible intervals
            theta_v_min = ci_theta[:, 0]
            theta_v_max = ci_theta[:, 1]
            # errorbars are the differences between x1 and x2
            below_error = theta_map - theta_v_min
            upper_error = theta_v_max - theta_map
            errorbars = np.row_stack((below_error, upper_error))
            errorbars1 = errorbars[:, :2]
            errorbars2 = errorbars[:, 2:]
        else:
            errorbars1 = None
            errorbars2 = None
        savename1 = f"{self._savename}/V_and_sigma"
        savename2 = f"{self._savename}/h"
        names1 = ["V", "sigma"]
        plot_triple_bar(safename=savename1, name_list=names1, values1=theta_ref[0:2],
                        values2=theta_map[0:2], values3=theta_true[0:2], name1="Guess",
                        name2="MAP estimate", name3="Ground truth",
                        errorbars=errorbars1)
        h_names = []
        for i in range(theta_map.size - 2):
            h_names.append(f"h_{i}")
        plot_triple_bar(safename=savename2, name_list=h_names, values1=theta_ref[2:],
                        values2=theta_map[2:], values3=theta_true[2:], name1="Guess",
                        name2="MAP estimate", name3="Ground truth",
                        errorbars=errorbars2)

    def _uq_error_analysis(self):
        """
        :return: uqerr_f, uqtightness_f, uqerr_theta, uqtightness_theta
        """
        # Compute UQ error measures for f.
        uqerr_f, uqsize_f, uqtightness_f = self._uq_error_analysis_f()
        uqerr_theta, uqtightness_theta = self._uq_error_analysis_theta()
        return uqerr_f, uqtightness_f, uqsize_f, uqerr_theta, uqtightness_theta

    def _uq_error_analysis_f(self):
        f_true = self._f_true
        f_map = self._fitted_model.f_map
        ci_f = self._uq.ci_f
        if ci_f is not None:
            filter_f = self._uq.filter_f
            uqerr_f = self._uq_error(f_true, ci_f, filter=filter_f)
            uqtightness_f = self._uq_tightness(f_true, f_map, ci_f, filter=filter_f)
            ci_f_size = ci_f[:, 1] - ci_f[:, 0]
            uqsize_f = np.mean(ci_f_size)
        else:
            uqerr_f = -1
            uqtightness_f = -1
            uqsize_f = -1
        return uqerr_f, uqtightness_f, uqsize_f

    def _uq_error_analysis_theta(self):
        ci_theta = self._uq.ci_theta
        if ci_theta is not None:
            theta_map = self._fitted_model.theta_map
            theta_true = self._theta_true
            uqerr_theta = self._uq_error(x=theta_true, ci=ci_theta, regop=self._statmodel.P2, filter=self._uq.filter_theta)
            uqtightness_theta = self._uq_tightness(x1=theta_true, x2=theta_map, ci=ci_theta, regop=self._statmodel.P2, filter=self._uq.filter_theta)
        else:
            uqerr_theta = 0
            uqtightness_theta = 0
        return uqerr_theta, uqtightness_theta

    def _uq_error(self, x, ci, filter: FilterFunction, regop: RegularizationOperator=None):
        """
        Checks whether the local means of x are actually inside ci.
        :param x: array_like, shape (N,)
        :param ci: array_like, shape (N,2)
        :return: float
            The list of uq errors
        """
        if regop is None:
            regop = IdentityOperator(x.size)
        x_filtered = filter.enlarge(filter.evaluate(x))
        too_large = regop.fwd(x_filtered - ci[:, 1])
        too_small = regop.fwd(ci[:, 0] - x_filtered)
        err_plus = self._positive_part(too_large)
        err_minus = self._positive_part(too_small)
        error = np.linalg.norm(np.concatenate((err_plus, err_minus))) / np.linalg.norm(regop.fwd(x))
        return error

    def _uq_tightness(self, x1, x2, ci, filter: FilterFunction, regop: RegularizationOperator=None):
        if regop is None:
            regop = IdentityOperator(x1.size)
        x1_filtered = filter.enlarge(filter.evaluate(x1))
        x2_filtered = filter.enlarge(filter.evaluate(x2))
        diff = regop.fwd(np.abs((x1_filtered - x2_filtered)))
        ci_size = regop.fwd(np.abs((ci[:, 1] - ci[:, 0])))
        # we compute the vector of ratios (clipping to avoid division by zero)
        ci_ratios = ci_size / diff.clip(min=1e-10)
        # then, we compute the median tightness
        tightness = np.median(ci_ratios)
        return tightness

    def _rdm(self, f, theta):
        """
        Computes the relative data misfit of x_map
        :return: float
        """
        x = self._statmodel._parameter_map.x(f, theta)
        misfit = self._fitted_model._problem.q.fwd(self._fitted_model._problem.fun(*x))
        rdm = np.linalg.norm(misfit) / np.sqrt(self._fitted_model._problem.scale)
        return rdm

    def _relative_error(self, map, truth):
        return np.linalg.norm(map - truth) / np.linalg.norm(truth)

    def _scaled_error(self, map, truth, regop):
        return np.linalg.norm(regop.fwd(map - truth)) / np.linalg.norm(regop.fwd(truth))

    def _positive_part(self, x):
        """
        Returns the positive part of the vector x
        """
        return x.clip(min=0.)

    def _image(self, f):
        m_f = self._statmodel.m_f
        n_f = self._statmodel.n_f
        f_im = np.reshape(f, (m_f, n_f))
        return f_im


