
import numpy as np
from experiment_kit.test import TestResult

from uq4pk_fit.visualization import plot_theta, plot_distribution_function, plot_blobs


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
    # do visualization for all scales
    for scale, postfix in zip(scales, scale_postfixes):
        _create_plots_for_f(savedir, tr, scale, postfix)
    # finally, also make feature plot
    if tr.features is not None:
        plot_blobs(image=f_map_image, blobs=tr.features, savefile=savedir + "/features.png", vmax=scale1)


def _create_plots_for_f(savedir: str, tr: TestResult, vmax: float, postfix: str):
    def ploty_f(image: np.ndarray, savename: str, vmax: float = None, vmin: float = None):
        return plot_distribution_function(image=image, savefile=savename, vmax=vmax, vmin=vmin,
                                  ssps=tr.ssps, show=False)
    f_true_im = tr.image(tr.data.f_true)
    f_map_im = tr.image(tr.f_map)
    f_ref_im = tr.image(tr.data.f_ref)
    vmin = 0.   # assume that all images are nonnegative
    ploty_f(image=f_true_im, savename=f"{savedir}/truth{postfix}", vmax=vmax, vmin=vmin)
    ploty_f(image=f_map_im, savename=f"{savedir}/map{postfix}", vmax=vmax, vmin=vmin)
    ploty_f(image=f_ref_im, savename=f"{savedir}/ref{postfix}", vmax=vmax, vmin=vmin)

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
        ploty_f(image=ci_lower_im, savename=f"{savedir}/lower{postfix}", vmax=vmax, vmin=vmin)
        ploty_f(image=ci_upper_im, savename=f"{savedir}/upper{postfix}", vmax=vmax, vmin=vmin)
        ploty_f(image=ci_sizes_im, savename=f"{savedir}/size{postfix}", vmax=vmax, vmin=vmin)
        ploty_f(image=phi_true_im, savename=f"{savedir}/filtered_truth{postfix}", vmax=vmax, vmin=vmin)
        ploty_f(image=phi_map_im, savename=f"{savedir}/filtered_map{postfix}", vmax=vmax, vmin=vmin)


def plot_theta_v(savedir: str, tr: TestResult):
    """
    Assuming that V and sigma are not fixed...
    """
    theta_map = tr.theta_map
    theta_ref = tr.data.theta_guess
    theta_true = tr.data.theta_true
    ci_theta = tr.ci_theta
    if ci_theta is not None:
        errorbars1 = ci_theta[0:2]
        errorbars2 = ci_theta[2:]
        errorbars3 = ci_theta[-2:]
    else:
        errorbars1 = None
        errorbars2 = None
        errorbars3 = None
    savename1 = f"{savedir}/V_and_sigma"
    savename2 = f"{savedir}/h"
    names1 = ["V", "sigma"]
    plot_theta(savefile=savename1, names=names1, theta_guess=theta_ref[0:2],
                    theta_map=theta_map[0:2], theta_true=theta_true[0:2],
                    ci_theta=errorbars1, show=False)
    h_names = []
    for i in range(theta_map.size - 2):
        h_names.append(f"h_{i}")
    plot_theta(savefile=savename2, names=h_names, theta_guess=theta_ref[2:],
                    theta_map=theta_map[2:], theta_true=theta_true[2:],
                    ci_theta=errorbars2, show=False)
    if theta_map.size == 7:
        savename3 = f"{savedir}/h_3_4"
        plot_theta(savefile=savename3, names=["h_3", "h_4"], theta_guess=theta_ref[-2:],
                        theta_map=theta_map[-2:], theta_true=theta_true[-2:], ci_theta=errorbars3, show=False)