
import numpy as np

from .forward_operator import ForwardOperator
from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction


class LightWeightedForwardOperator:

    def __init__(self, forward_operator: ForwardOperator, y: np.ndarray, theta: np.ndarray):
        """
        Creates a light-weighted forward operator G_bar based on a forward operator G, given by
        G_bar_j = G_j * sum(y) / sum(G_j),
        where G_j is the j-th column of G and G_bar_j is the j-th column of G_bar.

        :param forward_operator: The unnormalized forward operator G.
        :param y: The measurement.
        """
        # Check that the dimensions of G and y match
        assert y.shape == (forward_operator.dim_y, )

        self.dim_y = y.size
        self.dim_theta = forward_operator.dim_theta

        # Load G into a matrix.
        f0 = np.zeros(forward_operator.m_f * forward_operator.n_f)
        g = forward_operator.jac(f0, theta)[:, :-self.dim_theta]

        # Rescale the matrix.
        column_sums = np.sum(g, axis=0)
        assert column_sums.size == g.shape[1]
        y_sum = np.sum(y)
        weights = y_sum / column_sums
        g_bar = weights * g

        # Check that scaling worked.
        new_column_sums = np.sum(g_bar, axis=0)
        assert np.isclose(new_column_sums, y_sum, rtol=0.01).all()

        self._weights = weights
        self._g_bar = g_bar

    def fwd(self, f, theta):
        """
        :param f: array_like, (N,)
        :param theta: (K,)
        :return: array_like, (M,)
        """
        return self._g_bar @ f

    def jac(self, f, theta):
        dy_df = self._g_bar
        dy_dtheta = np.zeros((self.dim_y, self.dim_theta))
        dydx = np.concatenate((dy_df, dy_dtheta), axis=1)
        return dydx

    @property
    def weights(self) -> np.ndarray:
        """
        The weights d. We have
            G_bar = d * G, and f_bar = f / d.
        """
        return self._weights