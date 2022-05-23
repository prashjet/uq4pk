

from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.downsampling.RectangularDownsampling import RectangularDownsampling


def test_linear_interpolation():
    m = 12
    n = 53
    a = 2
    b = 2
    # Get test image.
    image = np.loadtxt("test_image.csv", delimiter=",")
    # Downsample the image.
    downsampler = RectangularDownsampling(shape=(m, n), a=a, b=b)
    u = downsampler.downsample(image.flatten())
    # Show downsampled image.
    m_a = np.ceil(m / a).astype(int)
    n_b = np.ceil(n / b).astype(int)
    image_downsampled = np.reshape(u, (m_a, n_b))
    # Then apply linear interpolation.
    image_interpolated = downsampler.enlarge(u=u.reshape(1, -1)).reshape(m, n)
    # Compare original image with interpolated one.
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(image, cmap="gnuplot")
    ax[1].imshow(image_downsampled, cmap="gnuplot")
    ax[2].imshow(image_interpolated, cmap="gnuplot")
    plt.show()
