"""
Creates plots for SVD-MCMC section.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from uq4pk_src import model_grids
from ..plot_params import CW
from .parameters import TIMES, QLIST, DIVERGENCES

# Use consistent style.
plt.style.use("src/uq4pk.mplstyle")

singval_plot = "SVD_MILES.png"
qtest_plot = "q_plot.png"


def plot_svd_mcmc(src: Path, out: Path):
    # Make plot of singular values.
    make_singular_values_plot(src, out)

    # Make plot of convergence wrt. q.
    #make_q_plot(src, out)


def make_singular_values_plot(src, out):
    ssps = model_grids.MilesSSP(lmd_min=5000, lmd_max=5500)
    light_weighted_ssps = ssps.X / np.sum(ssps.X, 0)
    U, S, VT = np.linalg.svd(light_weighted_ssps)
    fig, ax = plt.subplots(1, 1, figsize=(CW, 0.75 * CW))
    ax.semilogy(S)
    ax.axvline(15, ls=':', color='k')
    ax.set_xlabel('q')
    ax.set_ylabel('Singular value')
    ax.set_title('SVD of MILES templates')

    fig.tight_layout()
    fig.savefig(str(out / singval_plot), bbox_inches="tight")


def make_q_plot(src: Path, out: Path):
    # First, get relevant data.
    q_list = QLIST
    times = np.load(str(src / TIMES))
    divergences = np.load(str(src / DIVERGENCES))

    # If dimensions do not match, just make up q_list (this happens in test_mode).
    if q_list.size != times.size:
        print("WARNING: Size of q_list is wrong. Are you in test mode?")
        q_list = range(times.size)

    # Make two plots.
    fig, ax = plt.subplots(1, 2, figsize=(CW, 0.4 * CW))

    # First plot is divergences.
    ax[0].plot(q_list, divergences)
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("KL divergence")

    # Second plot is times.
    ax[1].plot(q_list, times)
    ax[1].set_xlabel("q")
    ax[1].set_ylabel("time")

    # Save figure.
    plt.savefig(str(out / qtest_plot))