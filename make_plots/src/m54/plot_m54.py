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
from ..plot_params import CW, CW2
from .parameters import RTHRESH1, RTHRESH2, \
    OVERLAP1, OVERLAP2, SIGMA_LIST, MEAN_SVDMCMC, LOWER_STACK_SVDMCMC, UPPER_STACK_SVDMCMC, MARGINAL_HMC, \
    MARGINAL_SVDMCMC, GROUND_TRUTH, PPXF, DATA, YMEAN_SVDMCMC, MASK, PREDICTIVE_SVDMCMC, MEAN_HMC,\
    YMEAN_HMC, LOWER_STACK_HMC, UPPER_STACK_HMC, PREDICTIVE_HMC, REAL1_NAME, REAL2_NAME, MAP_FILE, AGE_HMC, AGE_SVDMCMC
from ..util import add_colorbar_to_axis, add_colorbar_to_plot

plt.style.use("src/uq4pk.mplstyle")

# Define plot names.
m54_blobs_name = "_blobs.png"
m54_age_marginals_name = "_age.png"
m54_predictive_name = "_predictive.png"


def plot_m54(src: Path, out: Path):
    for dir, compare in zip([REAL1_NAME, REAL2_NAME], [True, False]):
        _m54_real_data_plot(src, out, dir=dir, with_comparison=compare)
        _m54_age_marginals_plot(src, out, dir=dir)
        _m54_predictive_plot(src, out, dir=dir)


def _m54_real_data_plot(src, out, dir: str, with_comparison: bool):
    """
    Produces figure 13 in the paper,
    under the name m54_real_data
    """
    # Get path to results for real data.
    real = src / dir
    too_young = 7
    # Get ground truth.
    ground_truth = np.load(str(real / GROUND_TRUTH))[:, too_young:]
    # Get ppxf fit.
    ppxf = np.load(str(real / PPXF))[:, too_young:]
    # Get MAP
    f_map = np.load(str(real / MAP_FILE))
    # Get vmax
    vmax = f_map.max()
    # Need correct SSPS grid.
    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
        age_lim=(0.1, 14)
    )
    # Adjust x-ticks.
    ssps.set_tick_positions([0.1, 1, 5, 13])

    ref_svdmcmc = f_map
    ref_hmc = f_map
    significant_blobs_svdmcmc, significant_blobs_hmc = \
        _get_blobs(src=real, mean_svdmcmc=ref_svdmcmc, mean_hmc=ref_hmc)

    # Create plot.
    if with_comparison:
        fig, ax = plt.subplots(2, 2, figsize=(CW2, 0.5 * CW2))
        plot_distribution_function(ax[0, 0], image=ground_truth, ssps=ssps, xlabel=False, ylabel=True)
        ax[0, 0].set_title("Resolved star counts")
        plot_distribution_function(ax[0, 1], image=ppxf, ssps=ssps, xlabel=False, ylabel=False)
        ax[0, 1].set_title("pPXF fit")
        i = 1
    else:
        fig, ax = plt.subplots(1, 2, figsize=(CW2, 0.25 * CW2))
        ax = ax.reshape((1, 2))
        i = 0

    plot_significant_blobs(ax=ax[i, 0], image=ref_svdmcmc, blobs=significant_blobs_svdmcmc, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=True)
    ax[i, 0].set_title("SVD-MCMC")
    im = plot_significant_blobs(ax=ax[i, 1], image=ref_hmc, blobs=significant_blobs_hmc, ssps=ssps, vmax=vmax,
                           xlabel=True, ylabel=False)
    ax[i, 1].set_title("Full MCMC")
    # Add colorbar to second image.
    add_colorbar_to_plot(fig, ax, im)
    plt.savefig(str(out / str(dir + m54_blobs_name)), bbox_inches="tight")


def _m54_age_marginals_plot(src, out, dir: str):
    """
    Creates figure 14 in the paper.
    """
    # Get results for real data.
    real = src / dir
    f_map = np.load(str(real / MAP_FILE))
    age_svdmcmc = np.load(str(real / AGE_SVDMCMC))
    age_hmc = np.load(str(real / AGE_HMC))
    age_marginal_svdmcmc = np.load(str(real / MARGINAL_SVDMCMC))
    age_marginal_hmc = np.load(str(real / MARGINAL_HMC))

    estimates = [age_svdmcmc, age_hmc]
    estimate_names = ["Posterior mean", "Posterior mean"]
    marginals = [age_marginal_svdmcmc, age_marginal_hmc]
    uppers = [marginal[0].max() for marginal in marginals]
    vmax = max(uppers)
    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
        age_lim=(0.1, 14)
    )
    # Adjust x-ticks.
    ssps.set_tick_positions([0.1, 1, 5, 13])

    # Create plot.
    fig, axes = plt.subplots(1, 2, figsize=(CW, 0.4 * CW))
    titles = ["SVD-MCMC", "Full MCMC"]
    for estimate, estimate_name, marginal, ax, title in zip(estimates, estimate_names, marginals, axes, titles):
        upper, lower = marginal
        # Visualize result.
        n = upper.size
        ax.set_ylim(0, vmax)
        x_span = np.arange(start=0, stop=1, step=(1 / n))
        # First, plot without strings.
        plt.rc('text', usetex=True)
        ax.plot(x_span, lower, color="b", linestyle="--", label="lower")
        ax.plot(x_span, upper, color="b", linestyle="-.", label="upper")
        ax.plot(x_span, estimate, label=estimate_name, color="r")
        ax.set_xlabel("Age [Gyr]")
        # Correct the ticks on x-axis.
        ax.set_xticks(ssps.img_t_ticks)
        ax.set_xticklabels(ssps.t_ticks)
        ax.set_title(title)
        # Shade area between lower1d and upper1d
        ax.fill_between(x_span, lower, upper, alpha=0.2)

    # First plot gets y-label.
    axes[0].set_ylabel("Density")
    plt.savefig(str(out / str(dir + m54_age_marginals_name)), bbox_inches="tight")


def _m54_predictive_plot(src, out, dir: str):
    """
    Create figure 16 in paper.
    """
    # Get results for real data.
    real = src / dir
    y = np.load(str(real / DATA))
    y[-1] = y[-2]  # Last value is NaN, substitute with second-to-last.
    y_mean_svdmcmc = np.load(str(real / YMEAN_SVDMCMC))
    ci_svdmcmc = np.load(str(real / PREDICTIVE_SVDMCMC))
    y_mean_hmc = np.load(str(real / YMEAN_HMC))
    ci_hmc = np.load(str(real / PREDICTIVE_HMC))
    mask = np.load(str(real / MASK))

    # Make plots.
    fig, axes = plt.subplots(2, 2, figsize=(CW2, 0.5 * CW2), gridspec_kw={"height_ratios": [1, 1]})
    # Posterior predictive plot for SVD-MCMC.
    _posterior_predictive_plot(ax1=axes[0, 0], ax2=axes[1, 0], y=y, y_est=y_mean_svdmcmc, mask=mask, ci=ci_svdmcmc)
    # Posterior predictive plot for HMC.
    _posterior_predictive_plot(ax1=axes[0, 1], ax2=axes[1, 1], y=y, y_est=y_mean_hmc, mask=mask, ci=ci_hmc)
    # Set axis labels and ticks.
    axes[0, 0].set_ylabel("Flux")
    axes[1, 0].set_ylabel("Residual [\%]")
    axes[0, 0].set_title("SVD-MCMC")
    axes[0, 1].set_title("Full MCMC")

    plt.savefig(str(out / str(dir + m54_predictive_name)), bbox_inches="tight")


def _posterior_predictive_plot(ax1, ax2, y: np.ndarray, y_est: np.ndarray, mask: np.ndarray, ci: np.ndarray):
    """
    Auxiliary function for making the posterior predictive plot.

    Parameters
    ----------
    ax1
        Axis where the data fit is plotted.
    ax2
        Axis where the residual is plotted.
    y
        The measured spectrum.
    y_est
        The predicted spectrum.
    mask
        Boolean mask that is False at values that weren't used in the inference.
    ci : shape (2, m)
        Upper and lower bounds of the simultaneous posterior predictive credible intervals.
    """
    lw = 0.2    # linewidth for everything
    # Get wavelengths.
    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)
    lmd = np.exp(m54_data.w)

    # Compare _y to y_med, with mask grayed out.
    ax1.plot(lmd, y, color="black", label="data", linewidth=lw)
    ax1.plot(lmd, y_est, color="red", label="fit", linewidth=lw)
    _gray_out_mask(ax=ax1, lmd=lmd, mask=mask)
    # Plot residual in %, with mask grayed out.
    assert np.all(y > 0.), "Cannot plot relative residual when '_y' is non-positive."
    # Residuals, ub_res and lb_res are each multiplied by 100 to get percentage values
    res = (y - y_est) / y * 100
    ub, lb = ci
    ub_res = (y - ub) / y * 100
    lb_res = (y - lb) / y * 100
    ax2.plot(lmd, res, color="black", linewidth=0.2, label="residual")
    ax2.fill_between(lmd, lb_res, ub_res, alpha=.3, color="blue")
    # Also plot horizontal line.
    ax2.axhline(0., linestyle="--", color="black")
    # Here, gray out masked values aswell.
    _gray_out_mask(ax=ax2, lmd=lmd, mask=mask)
    # Configure scale of _y-axis (hand-tuned).
    res_abs_max = np.max(np.abs(res))
    vmaxabs = 1.5 * res_abs_max
    vmax = vmaxabs
    vmin = -vmaxabs
    ax2.set_ylim(vmin, vmax)
    ax2.set_xlabel(r"Wavelength [$\textrm{\AA}$]")


def _get_blobs(src, mean_svdmcmc, mean_hmc):
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
                                                         overlap2=OVERLAP2,
                                                         exclude_max_scale=True)

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

    return significant_blobs_svdmcmc, significant_blobs_hmc


def _gray_out_mask(ax, lmd, mask):
    """
    Grays out the masked values in the given plot.
    """
    masked_indices = np.where(mask == False)[0]
    for i in masked_indices:
        gray_start = lmd[i]
        if i == lmd.size - 1:
            gray_end = lmd[i]
        else:
            gray_end = lmd[i + 1]
        ax.axvspan(gray_start, gray_end, facecolor="gray", alpha=0.2)

