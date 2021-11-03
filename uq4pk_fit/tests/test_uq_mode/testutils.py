
import uq4pk_fit.uq_mode as uq_mode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import scipy.linalg as scilin
from termcolor import colored



class TestProblem:
    def __init__(self, x_map: np.ndarray, x_true: np.ndarray, model: uq_mode.LinearModel, optimization_problem):
        self.x_map = x_map
        self.x_true = x_true
        self.model = model
        self.optimization_problem = optimization_problem

    def cost(self, x):
        return self.model.cost(x)

    def cost_grad(self, x):
        return self.model.cost_grad(x)


def check_pci(pci_array):
    for i in range(pci_array.shape[0]):
        if pci_array[i, 0] > pci_array[i, 1]:
            print(colored("Something is wrong with pci.", "red"))
            print(f"{pci_array[i, 0]} < {pci_array[i, 1]}?")


def get_points(ci_array, x_map):
    # the two columns of x_pci_0 constitute the endpoints of the 0-th projected credible interval
    x_pci_0 = np.row_stack((ci_array[0, :], x_map[1] * np.ones(2)))
    # correspondingly, the two columns of x_pci_1 constitute the endpoints of the 1-st projected credible interval
    x_pci_1 = np.row_stack((x_map[0] * np.ones(2), ci_array[1, :]))
    return x_pci_0, x_pci_1


def circle_points(center=(0, 0), r=1., n=100):
    """
    Generates 'n' points that are uniformly distributed on a circle with center 'center'
    and radius 'r'.
    """
    points = [
        (
            center[0] + (math.cos(2 * math.pi / n * x) * r),  # x
            center[1] + (math.sin(2 * math.pi / n * x) * r)  # y

        ) for x in range(0, n + 1)]
    return np.array(points).T


def matrix_inv_sqrt(mat):
    evals, evecs = scilin.eigh(mat)
    s = evecs * np.divide(1, np.sqrt(evals))
    return s


def plot_result(name, x_true, x_map, xi = None, boundary2=None, x_lci = None, x_pci = None, samples=None,
                lb=None):
    """
    Does some plotting, and also saves the figure under 'name.png'.
    """
    # VISUALIZE
    fig = plt.figure(0)
    if lb is None:
        lb = -np.inf*np.ones(2)
    if xi is not None:
        # plot square centered at MAP with side length 2 xi
        plt.gca().add_patch(Rectangle(x_map - xi, 2 * xi, 2 * xi, edgecolor="green", facecolor="none", lw=0.5,
                                      linestyle="--"))
    if boundary2 is not None:
        boundary2 = boundary2.clip(min=lb[:, np.newaxis])
        plt.plot(boundary2[0, :], boundary2[1, :], 'b--', ms=2, label='95%-Credible region')
    if samples is not None:
        plt.plot(samples[0, :], samples[1, :], 'rx', ms=1, label="Posterior samples")
    plt.plot(x_map[0], x_map[1], 'bo', label='MAP estimate')
    plt.plot(x_true[0], x_true[1], 'ko', label="True value")
    if x_pci is not None:
        x_pci_0, x_pci_1 = x_pci
        plt.plot(x_pci_0[0, :], x_pci_0[1, :], 'r--|', label="Pixelwise credible intervals")
        plt.plot(x_pci_1[0, :], x_pci_1[1, :], 'r--_')
    if x_lci is not None:
        x_lci_0, x_lci_1 = x_lci
        plt.plot(x_lci_0[0, :], x_lci_0[1, :], 'g-|', label='Local credible intervals')
        plt.plot(x_lci_1[0, :], x_lci_1[1, :], 'g-_')
    plt.axhline(y=0, color='k', lw=0.5)
    plt.axvline(x=0, color='k', lw=0.5)
    plt.legend(loc='upper right')
    plt.axis('scaled')
    fig.set_size_inches(10, 10)
    plt.savefig(f'{name}.png', bbox_inches="tight")
    plt.show()


def credible_region(alpha, H, y, Q, xbar, xmap, x_xi):
    """
    Computes the level that determines the (1-alpha) Pereyra credible region
    ||A(x-h)||_2^2 <= lvl for the linear Gaussian model
    Y = H @ X + V,
    V ~ normal(0, delta^2*Identity),
    X ~ normal(xbar, Identity).
    returns A, z, lvl
    """
    n = xmap.size
    tau = math.sqrt(16 * math.log(3/alpha) / n)
    map_cost = 0.5 * ((np.linalg.norm(Q @ (H @ xmap - y)))**2 + np.linalg.norm(xmap - xbar)**2)
    lvl = map_cost + n * (tau + 1)
    # completing the squares
    QH = Q @ H
    Qy = Q @ y
    A = 0.5 * (QH.T @ QH + np.identity(n))
    c = 0.5 * (Qy @ Qy + xbar @ xbar)
    b = - QH.T @ Qy - xbar
    s, U = scilin.eigh(A)
    A_sqrt_inv = U * np.divide(1, np.sqrt(s))
    A_inv = A_sqrt_inv @ A_sqrt_inv.T
    h = - 0.5 * A_inv @ b
    k = c - 0.25 * b.T @ A_inv @ b
    lvl -= k
    return A_sqrt_inv, h, lvl
