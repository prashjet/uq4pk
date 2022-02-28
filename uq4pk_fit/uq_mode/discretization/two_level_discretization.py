import numpy as np
from typing import Tuple

from ..geometry2d import coords_to_indices, indices_to_coords, rectangle_indices
from .rectangle_partition import rectangle_partition
from .image_discretization import ImageDiscretization, AdaptiveImageDiscretization


class TwoLevelDiscretization(ImageDiscretization):
    """
    A two-level discretization. Inside a rectangular window, the discretization is at the finest level.
    Outside, the discretization can be coarser.
    """
    def __init__(self, im_ref: np.ndarray, d1: int, d2: int, w1: int, w2: int, center: np.ndarray):
        """

        :param im_ref: Reference image.
        :param d1: The vertical resolution of the discretization. For example, if d1=2 and d2=3, the image is
            discretized into 2x3-rectangles (or smaller, if m and dim are not divisible by d1 and d2).
        :param d2: The horizontal resolution of the discretization.
        :param w1: The vertical radius of the window, IN MULTIPLES OF d1. For example, if d1=2 and w1=3, then the window
            will have height 2 * w1 * d1 + 1 = 13.
        :param w2: The horizontal radius of the window, IN MULTIPLES OF d2 (see w1).
        :param center: An array of the form (i, j), 0 <= i < m and 0 <= j < dim, denoting the pixel at which the window
            is centered.
        """
        # Check input for sensibleness.
        m, n = im_ref.shape
        assert d1 < m
        assert d2 < n
        assert 0 <= center[0] < m
        assert 0 <= center[1] < n

        # -- Set instance attributes.
        self.dim = m * n
        self.m = m
        self.n = n
        self._v = im_ref.flatten()
        # Make the basic partition into superpixels.
        partition = rectangle_partition(m=m, n=n, a=d1, b=d2)
        # Get coordinate of superpixel in which center is contained.
        index = coords_to_indices(m=m, n=n, coords=center)
        index_superpixel = partition.in_which_element(index)
        # Get superpixel-indices of inner window.
        m_down = np.floor(m / d1).astype(int)
        n_down = np.floor(n / d2).astype(int)
        active_superpixel_coord = indices_to_coords(m=m_down, n=n_down, indices=index_superpixel)
        diagonal_radius = np.array([w1, w2])
        upper_left = active_superpixel_coord - diagonal_radius
        lower_right = active_superpixel_coord + diagonal_radius
        superpixels_inside_window = rectangle_indices(m=m_down, n=n_down, upper_left=upper_left,
                                                      lower_right=lower_right)
        window_indices = np.concatenate([partition.get_element_list()[i] for i in superpixels_inside_window])
        # ensure that array has type int
        window_indices = window_indices.astype(int)
        self._window_indices = window_indices
        self.n_window = window_indices.size
        indices_of_superpixels_outside_window = list(set(np.arange(partition.size)) - set(superpixels_inside_window))
        self._superpixels_outside = [partition.get_element_list()[i] for i in indices_of_superpixels_outside_window]
        self.n_outside = len(self._superpixels_outside)
        self.dof = self.n_window + self.n_outside

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def u(self) -> np.ndarray:
        # Create u1
        u1 = np.zeros((self.dim, self.n_window))
        id_n_window = np.identity(self.n_window)
        u1[self._window_indices, :] = id_n_window
        # Create u2
        u2 = np.zeros((self.dim, self.n_outside))
        j = 0
        for superpixel in self._superpixels_outside:
            u2[superpixel, j] = 1
            j += 1
        u = np.concatenate([u1, u2], axis=1)
        return u

    def map(self, z) -> np.ndarray:
        x = np.zeros(self.dim)
        x[self._window_indices] = z[:self.n_window]
        j = 0
        z2 = z[self.n_window:]
        for superpixel in self._superpixels_outside:
            x[superpixel] = z2[j]
            j += 1
        return x + self._v

    def translate_lower_bound(self, lb: np.ndarray) -> np.ndarray:
        assert lb.size == self.dim
        lb_diff = lb - self._v
        lb_z1 = lb_diff[self._window_indices]
        lb_z2_list = []
        for superpixel in self._superpixels_outside:
            lb_z2_i = lb_diff[superpixel].max()
            lb_z2_list.append(lb_z2_i)
        lb_z2 = np.array(lb_z2_list)
        lb_z = np.concatenate([lb_z1, lb_z2])
        assert lb_z.size == self.dof
        return lb_z


class AdaptiveTwoLevelDiscretization(AdaptiveImageDiscretization):

    def __init__(self, im_ref: np.ndarray, d1: int, d2: int, w1: int, w2: int,):
        """
        :param im_ref: The reference image.
        :param d1: The vertical resolution of the discretization.
            For example, if d1=2 and d2=3, the image is discretized into 2x3-rectangles (or smaller, if m and dim are not
            divisible by d1 and d2).
        :param d2: The horizontal resolution of the discretization.
        :param w1: The vertical radius of the window, IN MULTIPLES OF d1. For example, if d1=2 and w1=3, then the window
            will have height 2 * w1 * d1 + 1 = 13.
        :param w2: The horizontal radius of the window, IN MULTIPLES OF d2 (see w1).
        """
        # - Check input
        m, n = im_ref.shape
        assert d1 < m
        assert d2 < n

        # - Set instance attributes
        self.m = m
        self.n = n
        self.dim = m * n
        # Set the discretizations
        all_indices = np.arange(self.dim)
        all_coords = indices_to_coords(m=m, n=n, indices=all_indices)
        discretization_list = []
        for coord in all_coords.T:
            window_centered_at_coord = TwoLevelDiscretization(im_ref=im_ref, d1=d1, d2=d2, w1=w1, w2=w2, center=coord)
            discretization_list.append(window_centered_at_coord)
        self.discretizations = discretization_list
