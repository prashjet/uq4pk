"""
Assembles the matrix representation of the Radon transform for given scaling factor.
"""


import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

from .params import SCALING


def assemble_radon(scaling_factor):
    """
    Assembles the matrix representation of a linear operator given by a function.
    :return: (m, dim)-ndarray
        The matrix A representing the operator.
    """
    image = rescale(shepp_logan_phantom(), scale=scaling_factor, mode='reflect', multichannel=False)
    n_1, n_2 = image.shape
    n = n_1 * n_2
    theta = np.linspace(0., 180., max(n_1, n_2), endpoint=False)
    def fwd(x):
        X = np.reshape(x, (n_1, n_2))
        y = radon(image=X, theta=theta, circle=False).flatten()
        return y
    a_list = []
    I_n = np.identity(n)
    print("Assembling Radon transform operator.")
    for j in range(n):
        print("\r", end="")
        print(f"{j+1}/{n}", end=" ")
        a_j = fwd(I_n[:, j])
        a_list.append(a_j)
    A = np.column_stack(a_list)
    # store it in a CSV file.
    np.savetxt(f'radon_{SCALING}.csv', A, delimiter=',')
    print("Assembly of the Radon forward operator completed.")
