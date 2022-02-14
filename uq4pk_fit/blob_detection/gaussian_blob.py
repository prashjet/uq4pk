
from math import sqrt
import numpy as np


class GaussianBlob:
    """
    Represents a two-dimensional anisotropic Gaussian blob.
    """
    def __init__(self, x: float, y: float, sigma_x: float, sigma_y: float, log: float, angle: float = 0):
        """
        We always use the axes
        0----> x
        |
        |
        v
        y
        (even if pyplot is too stupid to recognize this).

        :param x: Vertical position of the blob.
        :param y: Horizontal position of the blob.
        :param sigma_x: Vertical extend (standard deviation) of the blob.
        :param sigma_y: Horizontal extend (standard deviation) of the blob.
        :param log: The value of the Laplacian-of-Gaussian at the blob.
        :param angle: The blob's angle.
        """
        self._x = x
        self._y = y
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        self._log = log
        self._angle = angle

    @property
    def position(self) -> np.ndarray:
        """
        Returns the position [x, y] of the blob.
        """
        return np.array([self._x, self._y])

    @property
    def width(self) -> float:
        """
        The horizontal width of the blob. Since the horizontal radius r_y satisfies :math:`r_y = \\sqrt{2}\\sigma_y',
        the width, which is two-times the radius, is given by :math:'w = 2 \\sqrt{2} \\sigma_y`.
        """
        return 2 * sqrt(2) * self._sigma_y

    @property
    def angle(self) -> float:
        """
        The blob's angle (in degrees). The blob is rotated by 'angle' degrees in the counter-clockwise direction.
        """
        return self._angle

    @property
    def height(self) -> float:
        """
        The vertical height of the blob. It is given by 2 * sqrt(2) * sigma_x.
        """
        return 2 * sqrt(2) * self._sigma_x

    @property
    def vector(self) -> np.ndarray:
        """
        The vector representation [x, y, sigma_x, sigma_y] of the blob.
        """
        return np.array([self._x, self._y, self._sigma_x, self._sigma_y])

    @property
    def log(self) -> float:
        """
        The value of the blob's scale-space Laplacian.
        """
        return self._log