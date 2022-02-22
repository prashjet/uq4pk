
from matplotlib import pyplot as plt
import numpy as np
from skimage.data import coins

from uq4pk_fit.uq_mode.filter.gaussian_filter import GaussianFilterFunction2D, GaussianFilter2D

coin_image = coins()[10:80, 300:370]
m, n = coin_image.shape
sigma = np.array([1., 2.])

def test_gaussian_filter():
    center = np.array([20, 20])
    gaussian_filter = GaussianFilter2D(m=m, n=n, center=center, sigma=sigma, boundary="zero")

def test_gaussian_filter_function():
    gaussian_ff = GaussianFilterFunction2D(m=m, n=n, sigma=sigma, boundary="zero")
    coins_flat = coin_image.flatten()
    coins_smoothed = gaussian_ff.evaluate(coins_flat)

