import numpy as np
from scipy.linalg import block_diag

from .discretization import Discretization


class CombinedDiscretization(Discretization):
    """
    Combines two discretizations of spaces R^n1 and R^n2 into a combined discretization for the space R^(n1+n2).
    """
    def __init__(self, dis1: Discretization, dis2: Discretization):
        # The dimension of the combined discretization is simply the sum of the dimensions.
        self.dim = dis1.dim + dis2.dim
        # Same for the degrees of freedom.
        self.dof = dis1.dof + dis2.dof
        # Store the two sub-discretizations
        self._dis1 = dis1
        self._dis2 = dis2

    @property
    def u(self) -> np.ndarray:
        """
        The matrix U for the combined discretization is simply [U1, 0; 0, U2].
        """
        u1 = self._dis1.u
        u2 = self._dis2.u
        u = block_diag(u1, u2)
        assert u.shape == (self.dim, self.dof)
        return u

    @property
    def v(self) -> np.ndarray:
        """
        The vector v is simply [v1; v2].
        """
        v1 = self._dis1.v
        v2 = self._dis2.v
        return np.concatenate([v1, v2], axis=0)

    def map(self, z: np.ndarray) -> np.ndarray:
        """
        We have x(z) = [x(z1); x(z2)], for z = [z1; z2].
        """
        assert z.shape == (self.dof, )
        dim1 = self._dis1.dim
        z1 = z[:dim1]
        z2 = z[dim1:]
        x1 = self._dis1.map(z1)
        x2 = self._dis2.map(z2)
        x = np.concatenate([x1, x2], axis=0)
        assert x.shape == (self.dim, )
        return x

    def translate_lower_bound(self, lb: np.ndarray) -> np.ndarray:
        """
        lb_z = [lb_z1, lb_z2]
        """
        assert lb.shape == (self.dim, )
        dim1 = self._dis1.dim
        lb1 = lb[:dim1]
        lb2 = lb[dim1:]
        lb_z1 = self._dis1.translate_lower_bound(lb1)
        lb_z2 = self._dis1.translate_lower_bound(lb2)
        lb_z = np.concatenate([lb_z1, lb_z2])
        assert lb_z.shape == (self.dof, )
        return lb_z
