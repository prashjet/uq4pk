
from matplotlib import pyplot as plt
import numpy as np

from skimage.filters import gaussian
from uq4pk_fit.gaussian_blob import bessel2d
from uq4pk_fit.uq_mode import BesselFilterFunction2D
# Something is wrong with the Bessel functions


def test_bessel():
    f = np.load("m54_ground_truth.npy").reshape(12, -1)
    sigma = np.array([2, 4])
    f1 = gaussian(f, sigma=sigma, mode="reflect")
    f2 = bessel2d(f, sigma=sigma)
    bessel_filter = BesselFilterFunction2D(m=f.shape[0], n=f.shape[1], sigma=sigma, boundary="reflect")
    f3 = bessel_filter.evaluate(f.flatten()).reshape(f.shape)
    error_12 = np.linalg.norm(f1 - f2) / np.linalg.norm(f1)
    print(f"error_12 = {error_12}")
    error_23 = np.linalg.norm(f2 - f3) / np.linalg.norm(f2)
    print(f"error_23 = {error_23}")
    error_13 = np.linalg.norm(f1 - f3) / np.linalg.norm(f1)
    print(f"error_13 = {error_13}")
    fig, ax = plt.subplots(2, 2)
    vmax = f.max()
    ax[0, 0].imshow(f, vmax=vmax, vmin=0)
    ax[0, 1].imshow(f1, vmax=vmax, vmin=0)
    ax[1, 0].imshow(f2, vmax=vmax, vmin=0)
    ax[1, 1].imshow(f3, vmax=vmax, vmin=0)
    plt.show()

