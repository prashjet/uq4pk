import numpy as np

from ..geometry2d import indices_to_coords, rectangle_indices
from .image_discretization import AdaptiveImageDiscretization, ImageDiscretization


class Window(ImageDiscretization):
    """
    Given a pixel (i, j), this discretization lets only the pixels in a window of shape (2 * w1 + 1, 2 * w2 + 1),
    centered at the pixel (i,j), vary, where the window is cut-off at the boundaries.
    The rest of the image is fixed to a given reference image.
    """
    def __init__(self, im_ref: np.ndarray, center: np.ndarray, w1: int, w2: int):
        """

        :param im_ref: The reference image of shape (m, dim).
        :param center: Array of the form (i, j), with 0 <= i < m and 0 <= j < dim, describing the center pixel.
        :param w1: The vertical radius of the window, i.e. the window has a height of 2 * w1 + 1.
        :param w2: The horizontal radius of the window, i.e. the window has a width of 2 * w2 + 1.
        """
        # -- Check that input is correct
        assert im_ref.ndim == 2
        m, n = im_ref.shape
        assert 0 <= center[0] < m
        assert 0 <= center[1] < n

        # -- Set the attributes
        self.dim = im_ref.size
        self.m = m
        self.n = n
        self._v = im_ref.flatten()
        # Get the coordinates of the localization window
        diagonal = np.array([w1, w2])
        upper_left_corner = center - diagonal
        lower_right_corner = center + diagonal
        window_indices = rectangle_indices(m=m, n=n, upper_left=upper_left_corner, lower_right=lower_right_corner)
        # Translate the pixel-indices to flattened indices
        self._window_indices = window_indices
        self.dof = len(self._window_indices)

    @property
    def u(self) -> np.ndarray:
        """
        Returns the matrix U.
        """
        u = np.zeros((self.dim, self.dof))
        id_dof = np.identity(self.dof)
        u[self._window_indices, :] = id_dof
        return u

    @property
    def v(self):
        return self._v

    def map(self, z: np.ndarray) -> np.ndarray:
        """
        Computes x = U z + v
        """
        x = self._v
        x[self._window_indices] += z
        return x

    def translate_lower_bound(self, lb: np.ndarray) -> np.ndarray:
        assert lb.size == self.dim
        lb_z = (lb - self._v)[self._window_indices]
        assert lb_z.size == self.dof
        return lb_z


class LocalizationWindows(AdaptiveImageDiscretization):
    """
    An adaptive discretization, where each pixel is mapped to a localization window centered at that pixel.
    """
    def __init__(self, im_ref: np.ndarray, w1: int, w2: int):
        """

        :param im_ref: The reference image of shape (m, dim).
        :param w1: The vertical radius of the window, i.e. the window has a height of 2 * w1 + 1.
        :param w2: The horizontal radius of the window, i.e. the window has a width of 2 * w2 + 1.
        """
        # - Check input
        assert im_ref.ndim == 2

        # - Set instance attributes
        m, n = im_ref.shape
        self.m = m
        self.n = n
        self.dim = im_ref.size
        # Set the discretizations
        all_indices = np.arange(self.dim)
        all_coords = indices_to_coords(m=m, n=n, indices=all_indices)
        discretization_list = []
        for coord in all_coords.T:
            window_centered_at_coord = Window(im_ref=im_ref, w1=w1, w2=w2, center=coord)
            discretization_list.append(window_centered_at_coord)
        self.discretizations = discretization_list
