
import numpy as np
from typing import Sequence

from uq4pk_fit.blob_detection.scale_normalized_laplacian import scale_normalized_laplacian
from uq4pk_fit.blob_detection.scale_space_representation import scale_space_representation



def blob_operator(scales: Sequence[float], m: int, n: int, ratio: float = 1) -> np.ndarray:
    """
    Returns the matrix representation of the flattened blob operator
    :math:`\\Delta_x^h G_h: \\mathbb{R}^{mn} \\to \mathbb{R}^{smn}`,
    where s = len(scales).

    :param scales: The scale discretization.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param ratio: The height/width ratio for the Gaussian filter.
    :return: Array of shape (s*m*n, m * n), where s = len(scales).
    """
    # Assemble the discretization of :math:`\\nabla \\Delta_x^h G_h` by applying the operator to each basis element of
    # R^{mn}.
    basis = np.identity(m * n)
    out_list = []
    for basis_vector in basis:
        basis_image = np.reshape(basis_vector, (m, n))
        # Compute scale-space representation of basis image.
        f_x_h = scale_space_representation(basis_image, scales, mode="constant")
        out = scale_normalized_laplacian(ssr=f_x_h, scales=scales, mode="reflect").flatten()
        out_list.append(out)
    shape_operator = np.column_stack(out_list)
    assert shape_operator.shape == (len(scales) * m * n, m * n)
    return shape_operator