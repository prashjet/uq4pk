
from math import exp
import numpy as np
from scipy.stats import norm
from typing import Literal, Union

from ..geometry2d import indices_to_coords
from .filter_kernel import FilterKernel
from .kernel_filter import KernelFilter
from .image_filter_function import ImageFilterFunction
from scipy.special import iv


EPS = 1e-10


class BesselKernel2D(FilterKernel):
    """
    The two-dimensional Gaussian kernel k(x) = e^(-(x1/sqrt(2)*sigma)^2 - (x2/sqrt(2)*sigma)^2) / Z
    """
    def __init__(self, sigma: Union[float, np.ndarray]):
        """
        :param sigma: Standard deviation of the Gaussian kernel. Can also be a (2,) array, where the entries correspond
            to the standard deviations in the vertical and horizontal direction.
        """
        assert np.all(sigma >= EPS), "'sigma' must be strictly positive."
        self.dim = 2
        if isinstance(sigma, np.ndarray):
            assert sigma.shape == (2,)
            self._t1 = sigma[0] ** 2
            self._t2 = sigma[1] ** 2
        else:
            self._t1 = sigma ** 2
            self._t2 = sigma ** 2

    def weighting(self, x: np.ndarray) -> float:
        """
        :param x: Of shape (2, dim).
        :return: Vector of shape (dim, )
        """
        p1 = exp(-self._t1) * iv(x[0, :], self._t1)
        p2 = exp(-self._t1) * iv(x[1, :], self._t1)
        return p1 * p2


class BesselFilter2D(KernelFilter):

    def __init__(self, m: int, n: int, center: np.ndarray, sigma: Union[float, np.ndarray],
                 boundary: Literal["zero", "reflect"]):
        gaussian_kernel = BesselKernel2D(sigma=sigma)
        KernelFilter.__init__(self, m=m, n=n, center=center, kernel=gaussian_kernel, boundary=boundary)


class BesselFilterFunction2D(ImageFilterFunction):

    def __init__(self, m: int, n: int, sigma: Union[float, np.ndarray], boundary: Literal["zero", "reflect"]):
        """
        :param m: Image height in pixels.
        :param n: Image width in pixels.
        :param sigma: Standard deviation for Gaussian kernel.
        :param boundary: Determines how the image is extended at the boundaries.
        """
        # Prepare list of filters
        dim = m * n
        all_indices = np.arange(dim)
        all_coords = indices_to_coords(m=m, n=n, indices=all_indices)

        filter_list = []
        for ij in all_coords.T:
            filter_ij = BesselFilter2D(m=m, n=n, center=ij, sigma=sigma, boundary=boundary)
            filter_list.append(filter_ij)
        assert len(filter_list) == dim
        ImageFilterFunction.__init__(self, m, n, filter_list=filter_list)