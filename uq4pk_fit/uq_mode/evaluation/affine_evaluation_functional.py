
import numpy as np

class AffineEvaluationFunctional:
    """
    Abstract base class for evaluation functionals.
    An evaluation functional represents a way to compute a generalized credible intervals.
    Given a dependent variable :math:`z \idim \\mathbb{R}^k`, we have a linear loss function
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
    z0: np.ndarray  # Starting value for z for the optimization.
    w: np.ndarray   # The weight vector for the linear loss.

    @property
    def u(self) -> np.ndarray:
        """
        Returns the matrix U.
        """
        raise NotImplementedError

    @property
    def v(self) -> np.ndarray:
        """
        Returns the vector v
        """
        raise NotImplementedError

    def phi(self, z: np.ndarray) -> float:
        """
        The evaluation functional. The credible interval is understood to be [phi(z_min), phi(z_max)], where
        z_min is the minimizer of loss(z), and z_max is the maximizer.
        """
        raise NotImplementedError

    def x(self, z: np.ndarray) -> np.ndarray:
        """
        Shortcut for u @ z + v
        """
        raise NotImplementedError

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        """
        Function that transforms the lower bound on x to an equivalent lower bound on z.
        """
        raise NotImplementedError



