
import numpy as np

from .forward_operator import ForwardOperator
from .mass_weighted_forward_operator import MassWeightedForwardOperator


class LightWeightedForwardOperator(ForwardOperator):

    def __init__(self, theta: np.ndarray, ssps, dv=10, do_log_resample=True, hermite_order=4, mask=None):
        """
        Creates a light-weighted forward operator G_bar based on a forward operator G, given by
        G_bar_j = G_j * sum(y) / sum(G_j),
        where G_j is the j-th column of G and G_bar_j is the j-th column of G_bar.

        :param forward_operator: The unnormalized forward operator G.
        :param y: The measurement.
        """
        # Create a normal forward operator.
        mass_weigthed_fwdop = MassWeightedForwardOperator(ssps, dv, do_log_resample, hermite_order, mask)
        self.dim_theta = theta.size
        # Get the matrix representation at theta.
        self.m_f = mass_weigthed_fwdop.m_f
        self.n_f = mass_weigthed_fwdop.n_f
        f_test = np.ones((mass_weigthed_fwdop.m_f, mass_weigthed_fwdop.n_f))
        jac = mass_weigthed_fwdop.jac(f=f_test, theta=theta)
        x = jac[:, :-self.dim_theta]
        theta_jac = jac[:, -self.dim_theta:]
        self._theta_jac = theta_jac
        # normalize the sum of the columns.
        column_sums = np.sum(x, axis=0)
        # Divide by column sums.
        self._x_bar = x / column_sums[np.newaxis, :]

    def fwd(self, f, theta):
        """
        :param f: array_like, (N,)
        :param theta: (K,)
        :return: array_like, (M,)
        """
        return self._x_bar @ f

    def jac(self, f, theta):
        jac = np.column_stack([self._x_bar, self._theta_jac])
        return jac