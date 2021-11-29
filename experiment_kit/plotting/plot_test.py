
import numpy as np
from ..test import TestResult

from ..autodetect import detect
from .plot_triple_bar import plot_triple_bar
from .plot_with_colorbar import plot_with_colorbar


def plot_testresult(savedir: str, test_result: TestResult, extra_scale):
    """
    Creates plots from a TestResult object and saves them in the given location.
    """
    plot_f(savedir, test_result, extra_scale)
    plot_theta_v(savedir, test_result)


def plot_f(savedir: str, tr: TestResult, extra_scale=None):
    # make images
    f_true_image = tr.image(tr.data.f_true)
    f_map_image = tr.image(tr.f_map)
    f_ref_image = tr.image(tr.data.f_ref)
    f_all = np.concatenate((f_map_image.flatten(), f_ref_image.flatten(), f_true_image.flatten()))
    ci_f = tr.ci_f
    scale0 = None
    scale1 = f_all.max() # original scale
    scales = [scale0, scale1]
    if ci_f is not None:
        scale2 =  ci_f.max() # uq scale
        scales.append(scale2)
    scale_postfixes = ["", "_scale1", "_scale2"]
    if extra_scale is not None:
        scales.append(extra_scale)
        scale_postfixes.append("_rescaled")
    # do plotting for all scales
    for scale, postfix in zip(scales, scale_postfixes):
        _create_plots_for_f(savedir, tr, scale, postfix)
    # finally, also make feature plot
    if ci_f is not None:
        lower = tr.image(ci_f[:, 0])
        upper = tr.image(ci_f[:, 1])
        detect(map=f_map_image, lower=lower, upper=upper, scale=tr.uq_scale, savedir=savedir)

def _create_plots_for_f(savedir: str, tr: TestResult, vmax: float, postfix: str):
    def plot_f(image: np.ndarray, savename: str, vmax: float = None, vmin: float = None):
        return plot_with_colorbar(image=image, savename=savename, vmax=vmax, vmin=vmin,
                                  ticks=tr.ticks)
    f_true_im = tr.image(tr.data.f_true)
    f_map_im = tr.image(tr.f_map)
    f_ref_im = tr.image(tr.data.f_ref)
    vmin = 0.   # assume that all images are nonnegative
    plot_f(image=f_true_im, savename=f"{savedir}/truth{postfix}", vmax=vmax, vmin=vmin)
    plot_f(image=f_map_im, savename=f"{savedir}/map{postfix}", vmax=vmax, vmin=vmin)
    plot_f(image=f_ref_im, savename=f"{savedir}/ref{postfix}", vmax=vmax, vmin=vmin)

    ci_f = tr.ci_f
    if ci_f is not None:
        ci_upper = ci_f[:, 1]
        ci_upper_im = tr.image(ci_upper)
        ci_lower = ci_f[:, 0]
        ci_lower_im = tr.image(ci_lower)
        ci_sizes = ci_upper - ci_lower
        ci_sizes_im = tr.image(ci_sizes)
        phi_map_im = tr.image(tr.phi_map)
        phi_true_im = tr.image(tr.phi_true)
        plot_f(image=ci_lower_im, savename=f"{savedir}/lower{postfix}", vmax=vmax, vmin=vmin)
        plot_f(image=ci_upper_im, savename=f"{savedir}/upper{postfix}", vmax=vmax, vmin=vmin)
        plot_f(image=ci_sizes_im, savename=f"{savedir}/size{postfix}", vmax=vmax, vmin=vmin)
        plot_f(image=phi_true_im, vmax=vmax, vmin=vmin, savename=f"{savedir}/filtered_truth{postfix}")
        plot_f(image=phi_map_im, vmax=vmax, vmin=vmin, savename=f"{savedir}/filtered_map{postfix}")


def plot_theta_v(savedir: str, tr: TestResult):
    """
    Assuming that V and sigma are not fixed...
    """
    theta_map = tr.theta_map
    theta_ref = tr.data.theta_guess
    theta_true = tr.data.theta_true
    ci_theta = tr.ci_theta
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
    savename1 = f"{savedir}/V_and_sigma"
    savename2 = f"{savedir}/h"
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
    savename3 = f"{savedir}/h_3_4"
    if errorbars2 is None:
        reduced_errorbars = None
    else:
        reduced_errorbars = errorbars2[:, -2:]
    plot_triple_bar(safename=savename3, name_list=["h_3", "h_4"], values1=theta_ref[-2:],
                    values2=theta_map[-2:], values3=theta_true[-2:], name1="Guess",
                    name2="MAP estimate", name3="Ground truth",
                    errorbars=reduced_errorbars)