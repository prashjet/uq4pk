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
    """
    Creates figure A.1 in the paper (influence of latent dimension on SVD-MCMC).
    """
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
    # extract MC error
    mc_error = errors[-1]
    # Complement q_list
    q_list = np.append(q_list, [636])

    # If dimensions do not match, just make up q_list (this happens in test_mode).
    if q_list.size != times.size:
        print("WARNING: Size of q_list is wrong. Are you in test mode?")
        q_list = range(times.size)

    # First plot is error, measured in mean Jaccard distance.
    ax_left.semilogx(q_list, errors)
    ax_left.set_xlabel("q")
    ax_left.set_ylabel("Mean Jaccard distance")
    ymax = 1.1 * errors.max()
    ymax = max(ymax, mc_error)
    # Mark the q15-line.
    ax_left.axvline(15, ls=':', color='k')
    ax_left.text(18, 0.75, 'q=15', rotation=0, transform=ax_left.get_xaxis_text1_transform(0)[0], size="small")
    ax_left.set_ylim(0, ymax)

    # Second plot is times.
    ax_right.semilogx(q_list[:-1], times[:-1])
    ax_right.set_xlabel("q")
    ax_right.set_ylabel("Computation time [s]")
    # Enforce scientific notation on y-axis.
    ax_right.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Horizontal line for computation time of full MCMC.
    ax_right.axhline(times[-1], ls="--", color="b", lw=1)
    ymax = 1.1 * times.max()
    ax_right.axvline(15, ls=':', color='k')
    ax_right.text(18, 0.75, 'q=15', rotation=0, transform=ax_right.get_xaxis_text1_transform(0)[0], size="small")
    ax_right.set_ylim(0, ymax)

    # Save figure.
    fig.tight_layout()
    fig.savefig(str(out / plot_name), bbox_inches="tight")