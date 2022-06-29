

from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.downsampling.RectangularDownsampling import RectangularDownsampling
from uq4pk_fit.uq_mode.downsampling.downsampling1d import Downsampling1D


def test_linear_interpolation2d():
    m = 12
    n = 53
    a = 2
    b = 2
    # Get test image.
    image = np.loadtxt("test_image.csv", delimiter=",")
    # Downsample the image.
    downsampler = RectangularDownsampling(shape=(m, n), a=a, b=b)
    u = downsampler.downsample(image.flatten())
    m_a = np.ceil(m / a).astype(int)
    n_b = np.ceil(n / b).astype(int)
    image_downsampled = np.reshape(u, (m_a, n_b))
    # Then apply linear interpolation.
    image_interpolated = downsampler.enlarge(u=u.reshape(1, -1)).reshape(m, n)
    has_nans = np.any(np.isnan(image_interpolated))
    assert not has_nans
    # Compare original image with interpolated one.
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(image, cmap="gnuplot")
    ax[1].imshow(image_downsampled, cmap="gnuplot")
    ax[2].imshow(image_interpolated, cmap="gnuplot")
    plt.show()


def test_linear_interpolation1d():
    i = 6
    d = 3
    # Get test vector.
    image = np.loadtxt("test_image.csv", delimiter=",")
    x = image[i]
    n = x.size
    # Downsample the image.
    downsampler = Downsampling1D(n=n, d=d)
    u = downsampler.downsample(x)
    # Interpolate.
    x_hat = downsampler.enlarge(u.reshape(1, -1)).flatten()
    # Check for NaNs
    has_nans = np.any(np.isnan(x_hat))
    assert not has_nans
    # Compare in a plot.
    plt.figure(0)
    plt.plot(x, label="Original")
    plt.plot(x_hat, label="Interpolated")
    plt.legend()
    plt.show()

