
import numpy as np
from typing import List


class Discretization:
    """
    An affine discretization of R^dim is an affine map D: R^d -> R^dim, D(z) = U z + v, for z in R^d.
    The number d corresponds to the degrees of freedom of the discretization.
    Furthermore, each discretization has to define a way in which a lower bound x >= lb is translated to a lower bound
    z >= lb_z.
    """
    dim: int        # The dimension dim of the space.
    dof: int        # The degrees of freedom of the discretization.

    @property
    def u(self) -> np.ndarray:
        """
        Returns the matrix u.
        """
        raise NotImplementedError

    @property
    def v(self) -> np.ndarray:
        """
        Returns the vector v.
        """
        raise NotImplementedError

    def map(self, z: np.ndarray) -> np.ndarray:
        """
        Computes x = U z + v. (flattened)
        """
        raise NotImplementedError

    def translate_lower_bound(self, lb: np.ndarray) -> np.ndarray:
        """
        Translates a lower bound on x into a corresponding lower bound on z.
        """
        raise NotImplementedError


class AdaptiveDiscretization:
    """
    An adaptive discretization is a map that associates to each index i in [dim] a discretization of R^dim.
    """
    dim: int                                    # The dimension of the underlying space.
    discretizations: List[Discretization]       # Must have length 'dim'.


class AdaptiveDiscretizationFromList(AdaptiveDiscretization):

    def __init__(self, discretization_list: List[Discretization]):
        # Check that all discretizations have same dimension.
        dim1 = discretization_list[0].dim
        for dis in discretization_list:
            assert dis.dim == dim1

        # Initialize
        self.dim = dim1
        self.discretizations = discretization_list
