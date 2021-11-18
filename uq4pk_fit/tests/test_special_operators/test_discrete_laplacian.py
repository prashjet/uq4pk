
import cv2
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteLaplacian


def test_discrete_laplacian():
    testim = np.loadtxt("testim.csv", delimiter=",")
    testim_vec = testim.flatten()
    m, n = testim.shape
    # create discrete Laplacian operator
    laplacian = DiscreteLaplacian(m=m, n=n)
    filtered_vec = laplacian.fwd(testim_vec)
    filtered_image = np.reshape(filtered_vec, (m, n))
    plt.imshow(filtered_image)
    plt.show()