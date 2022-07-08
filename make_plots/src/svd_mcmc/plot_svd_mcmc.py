"""
Creates plots for SVD-MCMC section.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from uq4pk_src import model_grids
from ..plot_params import CW
from .parameters import TIMES, QLIST, ERRORS

# Use consistent style.
plt.style.use("src/uq4pk.mplstyle")

plot_name = "SVD.png"


def plot_svd_mcmc(src: Path, out: Path):

    fig = plt.figure(figsize=(CW, CW))
    ax_top = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
    ax_left = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
    ax_right = plt.subplot2grid(shape=(2, 2), loc=(1, 1))

    ssps = model_grids.MilesSSP(lmd_min=5000, lmd_max=5500)
    light_weighted_ssps = ssps.X / np.sum(ssps.X, 0)
    U, S, VT = np.linalg.svd(light_weighted_ssps)
    ax_top.semilogy(S)
    ax_top.axvline(15, ls=':', color='k')
    ax_top.set_xlabel('q')
    ax_top.set_ylabel('Singular value')
    ax_top.set_title('SVD of MILES templates')

    # First, get relevant data.
    q_list = QLIST
    times = np.load(str(src / TIMES))
    errors = np.load(str(src / ERRORS))

    # If dimensions do not match, just make up q_list (this happens in test_mode).
    if q_list.size != times.size:
        print("WARNING: Size of q_list is wrong. Are you in test mode?")
        q_list = range(times.size)

    # First plot is error, measured in mean Jaccard distance.
    ax_left.plot(q_list, errors)
    ax_left.set_xlabel("q")
    ax_left.set_ylabel("Error")
    ymax = 1.1 * errors.max()
    ax_left.set_ylim(0, ymax)

    # Second plot is times.
    ax_right.plot(q_list, times)
    ax_right.set_xlabel("q")
    ax_right.set_ylabel("Computation time")
    ymax = 1.1 * times.max()
    ax_right.set_ylim(0, ymax)

    # Save figure.
    fig.tight_layout()
    fig.savefig(str(out / plot_name), bbox_inches="tight")