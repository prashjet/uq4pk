"""
Some helper functions for the Radon demo
"""

from math import sqrt
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import skimage.data
import skimage.transform

from .params import BETA


def simulate_measurement(snr, scaling_factor, A):
    """
    Generates the Radon transform of the Shepp-Logan phantom and adds simulated noise with given signal-to-noise ratio
    :param snr: the desired signal-to-noise ratio
    :param scaling_factor: a scaling factor that determines the size of the y, and thereby also the parameter
    and measurement dimension
    :return: y_noisy, y, image, A, delta
        'y_noisy' is the noisy measurement (flat), 'y' is the original measurement, 'image' is the Shepp-Logan phantom,
        'A' is the matrix representation of the forward operator and
        'delta' is the standard deviation of the noise.
    """
    # load and rescale Shepp-Logan phantom
    shepp_logan = skimage.data.shepp_logan_phantom()
    image = skimage.transform.rescale(shepp_logan, scale=scaling_factor, mode='reflect', multichannel=False)
    x = image.flatten()
    n_1, n_2 = image.shape
    n = n_1 * n_2
    # compute the Radon transform of the Shepp-Logan phantom
    y = A @ x
    m = y.size
    # create a noisy measurement with the given signal-to-noise ratio
    delta = np.linalg.norm(y) / (snr * sqrt(m))
    standard_noise = np.random.randn(*y.shape)
    # rescale noise to approximately satisfy prescribed SNR
    noise = standard_noise * delta
    y_noisy = y + noise
    # rescale everything
    y_noisy = y_noisy
    y = y
    return y_noisy, y, image, delta


def plot_with_colorbar(image, vmax=None, vmin=None):
    colormap = plt.get_cmap("gnuplot")
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(image, cmap=colormap, vmax=vmax, vmin=vmin)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)


def compute_map(y, A, delta):
    """
    Given the forward operator A, the measurement y, and the standard deviation delta,
    computes the MAP estimator.
    :param y:
    :param A:
    :param delta:
    :return:
    """
    m = y.size
    G = A.T @ A + BETA * m * delta ** 2 * np.identity(A.shape[1])
    x_map = np.linalg.solve(G, A.T @ y)
    return x_map


def load_radon(scaling):
    radon_matrix = np.loadtxt(f"radon_{scaling}.csv", delimiter=",")
    return radon_matrix


def negative_log_posterior(x, y, A, delta):
    m = y.size
    misfit = 0.5 * (np.linalg.norm(y - A @ x) / delta) ** 2
    prior = 0.5 * BETA * m * np.linalg.norm(x) ** 2
    return misfit + prior


def negative_log_posterior_gradient(x, y, A, delta):
    m = y.size
    grad = BETA * m * x + A.T @ (A @ x - y) / (delta ** 2)
    return grad