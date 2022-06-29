"""
file: m54_plot.py
"""

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

import uq4pk_src
from uq4pk_fit.visualization import plot_distribution_function, plot_significant_blobs
from uq4pk_fit.blob_detection import detect_significant_blobs
from .parameters import MAP_FILE, LOWER_STACK_FILE, UPPER_STACK_FILE, RTHRESH1, RTHRESH2, \
    OVERLAP1, OVERLAP2, SIGMA_LIST, MEAN_SVDMCMC, LOWER_STACK_SVDMCMC, UPPER_STACK_SVDMCMC, MARGINAL_HMC, \
    MARGINAL_SVDMCMC, GROUND_TRUTH, PPXF, DATA, YMAP, YMEAN_SVDMCMC, MASK, PREDICTIVE_OPT, PREDICTIVE_SVDMCMC, MEAN_HMC,\
    YMEAN_HMC, LOWER_STACK_HMC, UPPER_STACK_HMC, MARGINAL_OPT, PREDICTIVE_HMC

from ..util import add_colorbar_to_axis, add_colorbar_to_plot


# Define plot names.
m54_real_data_name = "m54_real.png"
m54_mock_data_name = "m54_mock.png"
m54_age_marginals_name = "m54_age.png"
m54_predictive_name = "m54_predictive.png"


def plot_m54(src: Path, out: Path):
    _m54_real_data_plot(src, out)
    #_m54_mock_data_plot(src, out)
    _m54_age_marginals_plot(src, out)
    _m54_predictive_plot(src, out)


def _m54_real_data_plot(src, out):
    """
    Produces figure 13 in the paper,
    under the name m54_real_data
    """
    # Get path to results for real data.
    real = src / "m54_real"
    # Get ground truth.
    ground_truth = np.load(str(real / GROUND_TRUTH))
    # Get ppxf fit.
    ppxf = np.load(str(real / PPXF))
    # Get MAP and median.
    f_map = np.load(str(real / MAP_FILE))
    f_mean_svdmcmc = np.load(str(real / MEAN_SVDMCMC))
    f_mean_hmc = np.load(str(real / MEAN_HMC))
    # Get vmax
    vmax = max(f_map.max(), f_mean_hmc.max(), f_mean_svdmcmc.max())
    # Need correct SSPS grid.
    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
    )

    significant_blobs_svdmcmc, significant_blobs_opt, significant_blobs_hmc = \
        _get_blobs(src=real, mean_svdmcmc=f_mean_svdmcmc, map=f_map, mean_hmc=f_mean_hmc)

    # Create plot.
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 1), colspan=2)
    plot_distribution_function(ax1, image=ground_truth, ssps=ssps, xlabel=False, ylabel=True)
    ax2 = plt.subplot2grid(shape=(2, 6), loc=(0, 3), colspan=2)
    plot_distribution_function(ax2, image=ppxf, ssps=ssps, xlabel=False, ylabel=False)
    ax3 = plt.subplot2grid(shape=(2, 6), loc=(1, 0), colspan=2)
    plot_significant_blobs(ax=ax3, image=f_mean_svdmcmc, blobs=significant_blobs_svdmcmc, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=True)
    ax4 = plt.subplot2grid(shape=(2, 6), loc=(1, 2), colspan=2)
    im = plot_significant_blobs(ax=ax4, image=f_map, blobs=significant_blobs_opt, ssps=ssps, vmax=vmax,
                                xlabel=True, ylabel=False)
    ax5 = plt.subplot2grid(shape=(2, 6), loc=(1, 4), colspan=2)
    plot_significant_blobs(ax=ax5, image=f_mean_hmc, blobs=significant_blobs_hmc, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=False)
    # Add colorbar to second image.
    add_colorbar_to_axis(fig, ax=ax2, im=im)
    plt.savefig(str(out / m54_real_data_name), bbox_inches="tight")
    plt.show()


def _m54_mock_data_plot(src, out):
    """
    Creates figure 15 in the paper.
    """
    # Get path to results for real data.
    mock1 = src / "m54_mock1000"
    mock2 = src / "m54_mock100"
    # Get ground truth.
    ground_truth = np.load(str(mock1 / GROUND_TRUTH))
    # Get MAP and median.
    f_map1 = np.load(str(mock1 / MAP_FILE))
    f_mean_svdmcmc1 = np.load(str(mock1 / MEAN_SVDMCMC))
    f_mean_hmc1 = np.load(str(mock1 / MEAN_HMC))
    f_map2 = np.load(str(mock2 / MAP_FILE))
    f_mean_svdmcmc2 = np.load(str(mock2 / MEAN_SVDMCMC))
    f_mean_hmc2 = np.load(str(mock2 / MEAN_HMC))
    # Get vmax
    vmax = ground_truth.max()
    # Need correct SSPS grid.
    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
    )

    significant_blobs_svdmcmc1, significant_blobs_opt1, significant_blobs_hmc1 = \
        _get_blobs(src=mock1, mean_svdmcmc=f_mean_svdmcmc1, map=f_map1, mean_hmc=f_mean_hmc1)
    significant_blobs_svdmcmc2, significant_blobs_opt2, significant_blobs_hmc2 \
        = _get_blobs(src=mock2, mean_svdmcmc=f_mean_svdmcmc2, map=f_map2, mean_hmc=f_mean_hmc2)

    # Create plot.
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(10., 8.))
    # Plot ground truth over both upper windows.
    # Since it is on an unknown scale, we don't use the same vmax as for the reconstructions.
    ax1 = fig.add_subplot(gs[0, :])
    plot_distribution_function(ax1, image=ground_truth, ssps=ssps, xlabel=True, ylabel=True)
    # Plot results for high SNR.
    ax2 = fig.add_subplot(gs[1, 0])
    plot_significant_blobs(ax=ax2, image=f_mean_svdmcmc1, blobs=significant_blobs_svdmcmc1, ssps=ssps, vmax=vmax,
                           xlabel=False,
                           ylabel=True)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_significant_blobs(ax=ax3, image=f_map1, blobs=significant_blobs_opt1, ssps=ssps, vmax=vmax, xlabel=False,
                           ylabel=False)
    ax4 = fig.add_subplot(gs[1, 2])
    plot_significant_blobs(ax=ax4, image=f_mean_hmc1, blobs=significant_blobs_hmc1, ssps=ssps, vmax=vmax,
                           xlabel=False,
                           ylabel=False)
    # Plot results for low SNR.
    ax5 = fig.add_subplot(gs[2, 0])
    plot_significant_blobs(ax=ax5, image=f_mean_svdmcmc2, blobs=significant_blobs_svdmcmc2, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=True)
    ax6 = fig.add_subplot(gs[2, 1])
    immap = plot_significant_blobs(ax=ax6, image=f_map2, blobs=significant_blobs_opt2, ssps=ssps, vmax=vmax, xlabel=True,
                                   ylabel=False)
    ax7 = fig.add_subplot(gs[2, 2])
    plot_significant_blobs(ax=ax7, image=f_mean_hmc2, blobs=significant_blobs_hmc2, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=False)
    # Add colorbar to first image.
    add_colorbar_to_axis(fig, ax1, immap)
    # Save and show.
    plt.savefig(str(out / m54_mock_data_name), bbox_inches="tight")
    plt.show()


def _m54_age_marginals_plot(src, out):
    """
    Creates figure 14 in the paper.
    """
    # Get results for real data.
    real = src / "m54_real"
    f_map = np.load(str(real / MAP_FILE))
    f_mean_svdmcmc = np.load(str(real / MEAN_SVDMCMC))
    f_mean_hmc = np.load(str(real / MEAN_HMC))
    age_marginal_opt = np.load(str(real / MARGINAL_OPT))
    age_marginal_svdmcmc = np.load(str(real / MARGINAL_SVDMCMC))
    age_marginal_hmc = np.load(str(real / MARGINAL_HMC))

    estimates = [f_mean_svdmcmc, f_map, f_mean_hmc]
    estimate_names = ["Posterior mean", "MAP estimate", "Posterior mean"]
    marginals = [age_marginal_svdmcmc, age_marginal_opt, age_marginal_hmc]
    uppers = [marginal[0].max() for marginal in marginals]
    vmax = max(uppers)
    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
    )

    # Create plot.
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for estimate, estimate_name, marginal, ax in zip(estimates, estimate_names, marginals, axes):
        upper, lower = marginal
        # Marginalize estimate
        age_estimate = np.sum(estimate, axis=0)
        # Visualize result.
        n = upper.size
        ax.set_ylim(0, vmax)
        x_span = np.arange(start=0, stop=1, step=(1 / n))
        # First, plot without strings.
        plt.rc('text', usetex=True)
        ax.plot(x_span, lower, color="b", linestyle="--", label="lower")
        ax.plot(x_span, upper, color="b", linestyle="-.", label="upper")
        ax.plot(x_span, age_estimate, label=estimate_name, color="r")
        ax.set_xlabel("Age [Gyr]")
        # Correct the ticks on x-axis.
        ax.set_xticks(ssps.img_t_ticks)
        ax.set_xticklabels(ssps.t_ticks)
        plt.legend()
        # Shade area between lower1d and upper1d
        ax.fill_between(x_span, lower, upper, alpha=0.2)

    # First plot gets y-label.
    axes[0].set_ylabel("Density")
    plt.savefig(str(out / m54_age_marginals_name), bbox_inches="tight")
    plt.show()


def _m54_predictive_plot(src, out):
    """
    Create figure 16 in paper.
    """
    # Get results for real data.
    real = src / "m54_real"
    y = np.load(str(real / DATA))
    y[-1] = y[-2]  # Last value is NaN, substitute with second-to-last.
    y_map = np.load(str(real / YMAP))
    ci_opt = np.load(str(real / PREDICTIVE_OPT))
    y_mean_svdmcmc = np.load(str(real / YMEAN_SVDMCMC))
    ci_svdmcmc = np.load(str(real / PREDICTIVE_SVDMCMC))
    y_mean_hmc = np.load(str(real / YMEAN_HMC))
    ci_hmc = np.load(str(real / PREDICTIVE_HMC))
    mask = np.load(str(real / MASK))

    # Make plots.
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), gridspec_kw={"height_ratios": [1, 1]})
    # Posterior predictive plot for SVD-MCMC.
    _posterior_predictive_plot(ax1=axes[0, 0], ax2=axes[1, 0], y=y, y_est=y_mean_svdmcmc, mask=mask, ci=ci_svdmcmc)
    # Posterior predictive plot for optimization-based approach.
    _posterior_predictive_plot(ax1=axes[0, 1], ax2=axes[1, 1], y = y, y_est = y_map, mask = mask, ci = ci_opt)
    # Posterior predictive plot for HMC.
    _posterior_predictive_plot(ax1=axes[0, 2], ax2=axes[1, 2], y=y, y_est=y_mean_hmc, mask=mask, ci=ci_hmc)
    # Set axis labels and ticks.
    axes[0, 0].set_ylabel("Flux")
    axes[1, 0].set_ylabel("Residual [%]")

    plt.savefig(str(out / m54_predictive_name), bbox_inches="tight")
    plt.show()


def _posterior_predictive_plot(ax1, ax2, y: np.ndarray, y_est: np.ndarray, mask: np.ndarray, ci: np.ndarray):
    """
    Makes a posterior predictive plot.

    :param ax1: Axis where the data fit is plotted.
    :param ax2: Axis where the residual is plotted.
    :param y: (m, )-array. The real data.
    :param y_est: (m, )-array. The predicted data.
    :param mask: (m, )-array. Boolean mask that is False at values that weren't used in the estimation.
    :param ci: (2, m)-array. Upper and lower bounds of the simultaneous posterior predictive credible intervals.
    """
    lw = 0.2    # linewidth for everything
    # Get wavelengths.
    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)
    lmd = np.exp(m54_data.w)
    y_scale = np.sum(y)
    y_est_scale = np.sum(y_est)
    # Compare y to y_med, with mask grayed out.
    ax1.plot(lmd, y, color="black", label="data", linewidth=lw)
    ax1.plot(lmd, y_est, color="red", label="fit", linewidth=lw)
    masked_indices = np.where(mask == False)[0]
    for i in masked_indices:
        j = lmd[0] + i
        ax1.axvspan(j - 0.5, j + 0.5, facecolor="gray", alpha=0.2)
    # Plot residual in %, with mask grayed out.
    assert np.all(y > 0.), "Cannot plot relative residual when 'y' is non-positive."
    res = (y - y_est) / y
    ub, lb = ci
    ub_res = (y - ub) / y
    lb_res = (y - lb) / y
    ax2.plot(lmd, res, color="black", linewidth=0.2, label="residual")
    ax2.fill_between(lmd, lb_res, ub_res, alpha=.3, color="blue")
    # Also plot horizontal line.
    ax2.axhline(0., linestyle="--", color="black")
    # Here, gray out masked values aswell.
    for i in masked_indices:
        j = lmd[0] + i
        ax2.axvspan(j - 0.5, j + 0.5, facecolor="gray", alpha=0.2)
    # Configure scale of y-axis (hand-tuned).
    res_abs_max = np.max(np.abs(res))
    vmaxabs = 1.5 * res_abs_max
    vmax = vmaxabs
    vmin = -vmaxabs
    ax2.set_ylim(vmin, vmax)
    ax2.set_xlabel("Wavelength [nm]")


def _get_blobs(src, mean_svdmcmc, map, mean_hmc):
    # Perform ULoG based on MCMC.
    lower_stack_svdmcmc = np.load(str(src / LOWER_STACK_SVDMCMC))
    upper_stack_svdmcmc = np.load(str(src / UPPER_STACK_SVDMCMC))
    significant_blobs_svdmcmc = detect_significant_blobs(reference=mean_svdmcmc,
                                                         sigma_list=SIGMA_LIST,
                                                         lower_stack=lower_stack_svdmcmc,
                                                         upper_stack=upper_stack_svdmcmc,
                                                         rthresh1=RTHRESH1,
                                                         rthresh2=RTHRESH2,
                                                         overlap1=OVERLAP1,
                                                         overlap2=OVERLAP2)

    # Perform ULoG based on optimization.
    lower_stack_opt = np.load(str(src / LOWER_STACK_FILE))
    upper_stack_opt = np.load(str(src / UPPER_STACK_FILE))
    significant_blobs_opt = detect_significant_blobs(reference=map,
                                                     sigma_list=SIGMA_LIST,
                                                     lower_stack=lower_stack_opt,
                                                     upper_stack=upper_stack_opt,
                                                     rthresh1=RTHRESH1,
                                                     rthresh2=RTHRESH2,
                                                     overlap1=OVERLAP1,
                                                     overlap2=OVERLAP2)

    # Perform ULoG based on full HMC.
    lower_stack_hmc = np.load(str(src / LOWER_STACK_HMC))
    upper_stack_hmc = np.load(str(src / UPPER_STACK_HMC))
    significant_blobs_hmc = detect_significant_blobs(reference=mean_hmc,
                                                         sigma_list=SIGMA_LIST,
                                                         lower_stack=lower_stack_hmc,
                                                         upper_stack=upper_stack_hmc,
                                                         rthresh1=RTHRESH1,
                                                         rthresh2=RTHRESH2,
                                                         overlap1=OVERLAP1,
                                                         overlap2=OVERLAP2)

    return significant_blobs_svdmcmc, significant_blobs_opt, significant_blobs_hmc
