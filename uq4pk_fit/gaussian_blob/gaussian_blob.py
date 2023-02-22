
from math import sqrt
import numpy as np
from typing import Union


class GaussianBlob:
    """
    Represents a two-dimensional Gaussian blob.
    """
    def __init__(self, x1: int, x2: int, sigma: Union[float, np.ndarray], log: float):
        """
        We always use the axes
        0----> x2
        |
        |
        v
        x1
        (even if pyplot is too stupid to recognize this).
        Parameters
        ----------
        x1 : int
            The vertical position of the blob center.
        x2 : int
            The horizontal position of the blob center.
        sigma :
            The standard deviation associated to the blob.
        log :
            The Laplacian-of-Gaussians value associated to the blob.
        """
        self._x1 = x1
        self._x2 = x2
        if isinstance(sigma, np.ndarray):
            self._sigma1 = sigma[0]
            self._sigma2 = sigma[1]
        else:
            self._sigma1 = sigma
            self._sigma2 = sigma
        self._log = log

    @property
    def position(self) -> np.ndarray:
        """
        Returns the position [x, y] of the blob.
        """
        return np.array([self._x1, self._x2])

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def x2(self) -> int:
        return self._x2

    @property
    def width(self) -> float:
        """
        The horizontal width of the blob. Since the horizontal radius r_y satisfies :math:`r_y = \\sqrt{2}\\sigma_y',
        the width, which is two-times the radius, is given by :math:'w = 2 \\sqrt{2} \\sigma_y`.
        """
        return 2 * sqrt(2) * self._sigma2

    @property
    def height(self) -> float:
        """
        The vertical height of the blob. It is given by 2 * sqrt(2) * sigma_x.
        """
        return 2 * sqrt(2) * self._sigma1

    @property
    def vector(self) -> np.ndarray:
        """
        The vector representation [x, y, sigma_x, sigma_y] of the blob.
        """
        return np.array([self._x1, self._x2, self._sigma1, self._sigma2])

    @property
    def log(self) -> float:
        """
        The value of the blob's scale-space Laplacian.
        """
        return self._log

    @property
    def scale(self) -> float:
        """
        The scale of a blob with deviations [sigma_1, sigma_2] is defined as 0.5 * ||sigma||^2.
        """
        return 0.5 * (self._sigma1 ** 2 + self._sigma2 ** 2)
