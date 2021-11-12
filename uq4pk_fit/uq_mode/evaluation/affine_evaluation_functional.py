
import numpy as np

class AffineEvaluationFunctional:
    """
    Abstract base class for evaluation functionals.
    An evaluation functional represents a way to compute a generalized credible intervals.
    Given a dependent variable :math:`z \in \\mathbb{R}^k`, we have a linear loss function
    :math:`J: \\mathbb{R}^k \to \R', J(z) = w.T z,
     which is minimized/maximized, and an affine map x(z) = U z + v,
    that maps the dependent variable `z` to the parameter x.
    Furthermore, there is an evaluation function phi: R^k to R^l, which maps z to the quantity of interest that is then
    used as lower/upper bound for the credible interval.

    As an example, for (modified) local credible intervals, the dependent variable would be z = eta in R, the
    loss function would simply be J(z) = z, and for the affine map x(z) = M z + v, we would have M = zeta_I and
    v = x_map,
    where I is the
    corresponding index set. Finally, the evaluation map would be phi: R^k \to R^l, phi(z) = x_map_I + z, i.e. we have
    l = #I.
    """
    dim: int        # The dimension N of the parameter space.
    zdim: int       # The dimension of the dependent variable.
    phidim: int     # The dimension of the output of phi.
    w: np.ndarray   # The vector that determines the linear loss function J(z) = w^\top z.
    u: np.ndarray   # The matrix M in the affine map x(z) = U z + v
    v: np.ndarray   # The vector v in the affine map x(z) = U z + v
    z0: np.ndarray  # Starting value for z for the optimization.

    def phi(self, z: np.ndarray) -> np.ndarray:
        """
        The evaluation functional. The credible interval is understood to be [phi(z_min), phi(z_max)], where
        z_min is the minimizer of loss(z), and z_max is the maximizer.
        """
        raise NotImplementedError

    def x(self, z: np.ndarray) -> np.ndarray:
        """
        Shortcut for u @ z + v
        """
        return self.u @ z + self.v

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        """
        Function that transforms the lower bound on x to an equivalent lower bound on z.
        """
        raise NotImplementedError



