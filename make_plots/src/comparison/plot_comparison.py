"""
file: comparison_plot.py
"""

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from pathlib import Path

import uq4pk_src
from uq4pk_fit.visualization import plot_significant_blobs, plot_distribution_function
from uq4pk_fit.blob_detection import detect_significant_blobs
from src.comparison.parameters import SIGMA_LIST, MEDIANFILE, MAPFILE, LOWER_STACK_OPT, UPPER_STACK_OPT, \
    LOWER_STACK_MCMC, UPPER_STACK_MCMC, RTHRESH1, RTHRESH2, OVERLAP1, OVERLAP2, LMD_MIN, LMD_MAX, DV, OUT1, OUT2, \
    OUT3, TRUTHFILE

from ..util import add_colorbar_to_axis, add_colorbar_to_plot


demo_sigma = 3      # index of scale for which FCIs are plotted.
# Setup SSPS grid.
ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
ssps.logarithmically_resample(dv=DV)

# Set names for plots.
fci_name = "comparison_fcis.png"    # name for figure 11
blob_name = "comparison_blobs.png"  # name for figure 12


def plot_comparison(src: Path, out: Path):
    _compare_blobs(src, out)
    _compare_fci(src, out)


def _compare_fci(src: Path, out: Path):
    """
    Creates figure 11 in the paper.
    """
    # Currently, using out2.

    src_dir = src / OUT1

    # ----------- PLOT DIFFERENT FCIS

    # Load sampling-based stack
    lower_stack_sampled = np.load(str(src_dir / LOWER_STACK_MCMC))
    upper_stack_sampled = np.load(str(src_dir / UPPER_STACK_MCMC))
    # Load optimization-based stack
    lower_stack = np.load(str(src_dir / LOWER_STACK_OPT))
    upper_stack = np.load(str(src_dir / UPPER_STACK_OPT))
    # Select the correct FCIs.
    fci_low_mcmc = lower_stack_sampled[demo_sigma]
    fci_upp_mcmc = upper_stack_sampled[demo_sigma]
    fci_low_opt = lower_stack[demo_sigma]
    fci_upp_opt = upper_stack[demo_sigma]
    # And plot.
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    vmax = max(fci_upp_opt.max(), fci_upp_mcmc.max())
    im = plot_distribution_function(ax=axes[0, 0], image=fci_upp_mcmc, vmax=vmax, ssps=ssps, flip=False)
    plot_distribution_function(ax=axes[1, 0], image=fci_low_mcmc, vmax=vmax, ssps=ssps, flip=False)
    plot_distribution_function(ax=axes[0, 1], image=fci_upp_opt, vmax=vmax, ssps=ssps, flip=False)
    plot_distribution_function(ax=axes[1, 1], image=fci_low_opt, vmax = vmax, ssps = ssps, flip = False)

    # Add colorbar.
    add_colorbar_to_plot(fig, axes, im)

    plt.savefig(str(out / fci_name), bbox_inches="tight")
    plt.show()


def _compare_blobs(src: Path, out: Path):
    """
    Creates figure 12 in the paper.
    """
    # Get ground truth.
    f_true = np.load(str(src / OUT1 / TRUTHFILE)).reshape(12, 53)

    # Create plot.
    gs = gridspec.GridSpec(4, 2)
    fig = plt.figure(figsize=(10., 12.))
    # Plot ground truth, with colorbar.
    vmax = f_true.max()
    ax1 = fig.add_subplot(gs[0, :])
    im = plot_distribution_function(ax1, image=f_true, ssps=ssps, flip=False)
    # Add colorbar
    add_colorbar_to_axis(fig, ax1, im)

    # Create plots for each of the settings.
    for i, outi in enumerate([OUT1, OUT2, OUT3]):
        # Load point estimates.
        f_map = np.load(str(src / outi / MAPFILE))
        f_median = np.load(str(src / outi / MEDIANFILE))

        # Perform blob detection for MCMC.
        lower_stack_mcmc = np.load(str(src / outi / LOWER_STACK_MCMC))
        upper_stack_mcmc = np.load(str(src / outi / UPPER_STACK_MCMC))
        mcmc_blobs = detect_significant_blobs(sigma_list=SIGMA_LIST,
                                                 lower_stack=lower_stack_mcmc,
                                                 upper_stack=upper_stack_mcmc,
                                                 reference=f_median,
                                                 rthresh1=RTHRESH1,
                                                 rthresh2=RTHRESH2,
                                                 overlap1=OVERLAP1,
                                                 overlap2=OVERLAP2)
        # Perform blob detection for optimization.
        lower_stack_opt = np.load(str(src / outi / LOWER_STACK_OPT))
        upper_stack_opt = np.load(str(src / outi / UPPER_STACK_OPT))
        opt_blobs = detect_significant_blobs(sigma_list=SIGMA_LIST,
                                                 lower_stack=lower_stack_opt,
                                                 upper_stack=upper_stack_opt,
                                                 reference=f_map,
                                                 rthresh1=RTHRESH1,
                                                 rthresh2=RTHRESH2,
                                                 overlap1=OVERLAP1,
                                                 overlap2=OVERLAP2)
        # Visualize.
        ax_left = fig.add_subplot(gs[i + 1, 0])
        ax_right = fig.add_subplot(gs[i + 1, 1])
        plot_significant_blobs(ax=ax_left, image=f_median, blobs=mcmc_blobs, vmax=vmax, ssps=ssps, flip=False)
        plot_significant_blobs(ax=ax_right, image=f_map, blobs=opt_blobs, vmax=vmax, ssps=ssps, flip=False)

    plt.savefig(str(out / blob_name), bbox_inches="tight")
    plt.show()



def _comparison_plot_loc(src: Path, postfix: str, out: Path):
    # Names of plots
    map_plot = "comparison_map_" + postfix
    truth_plot = "comparison_truth_" + postfix
    median_plot = "comparison_posterior_median_" + postfix
    fci_low_opt_plot = "comparison_fci_low_opt_" + postfix
    fci_upp_opt_plot = "comparison_fci_upp_opt_" + postfix
    fci_low_mcmc_plot = "comparison_fci_low_mcmc_" + postfix
    fci_upp_mcmc_plot = "comparison_fci_upp_mcmc_" + postfix
    blobs_opt_plot = "comparison_significant_blobs_opt_" + postfix
    blobs_mcmc_plot = "comparison_significant_blobs_mcmc_" + postfix

    # ----------- LOAD POINT ESTIMATES (AND GROUND TRUTH)

    f_map = np.load(str(src / MAPFILE))
    f_median = np.load(str(src / MEDIANFILE))
    f_true = np.load(str(src / TRUTHFILE)).reshape(12, 53)

    # ----------- PLOT DIFFERENT FCIS

    # Load optimization-based stack
    lower_stack = np.load(str(src / LOWER_STACK_OPT))
    upper_stack = np.load(str(src / UPPER_STACK_OPT))
    # Load sampling-based stack
    lower_stack_sampled = np.load(str(src / LOWER_STACK_MCMC))
    upper_stack_sampled = np.load(str(src / UPPER_STACK_MCMC))
    # Select the correct FCIs.
    fci_low_opt = lower_stack[demo_sigma]
    fci_upp_opt = upper_stack[demo_sigma]
    fci_low_mcmc = lower_stack_sampled[demo_sigma]
    fci_upp_mcmc = upper_stack_sampled[demo_sigma]
    # And plot.
    vmax = max(fci_upp_opt.max(), fci_upp_mcmc.max())
    plot_distribution_function(image=fci_low_opt, vmax=vmax, savefile=str(out / fci_low_opt_plot))
    plot_distribution_function(image=fci_upp_opt, vmax=vmax, savefile=str(out / fci_upp_opt_plot))
    plot_distribution_function(image=fci_low_mcmc, vmax=vmax, savefile=str(out / fci_low_mcmc_plot))
    plot_distribution_function(image=fci_upp_mcmc, vmax=vmax, savefile=str(out / fci_upp_mcmc_plot))

    # ---------- BLOB DETECTION BASED ON CONVEX OPTIMIZATION

    vmax = max(f_true.max(), f_map.max(), f_median.max())
    plot_distribution_function(image=f_true, savefile=str(out / truth_plot), vmax=vmax)
    plot_distribution_function(image=f_map, savefile=str(out / map_plot), vmax=vmax)
    matched_blobs = detect_significant_blobs(sigma_list=SIGMA_LIST,
                                             lower_stack=lower_stack,
                                             upper_stack=upper_stack,
                                             reference=f_map,
                                             rthresh1=RTHRESH1,
                                             rthresh2=RTHRESH2,
                                             overlap1=OVERLAP1,
                                             overlap2=OVERLAP2)
    plot_significant_blobs(image=f_map, blobs=matched_blobs, savefile=str(out / blobs_opt_plot), ssps=ssps, vmax=vmax)

    # ---------- BLOB DETECTION BASED ON SVD-HMC

    plot_distribution_function(image=f_median, savefile=str(out / median_plot))
    matched_blobs_sampled = detect_significant_blobs(sigma_list=SIGMA_LIST,
                                                     lower_stack=lower_stack_sampled,
                                                     upper_stack=upper_stack_sampled,
                                                     reference=f_median,
                                                     rthresh1=RTHRESH1,
                                                     rthresh2=RTHRESH2,
                                                     overlap1=OVERLAP1,
                                                     overlap2=OVERLAP2)
    plot_significant_blobs(image=f_median, blobs=matched_blobs_sampled, savefile=str(out / blobs_mcmc_plot),
                           ssps=ssps, vmax=vmax)
