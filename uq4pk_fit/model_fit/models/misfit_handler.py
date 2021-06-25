
import numpy as np

from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction

class MisfitHandler:
    """
    handles the misfit, lol
    """
    def __init__(self, y, hermite_order):
        """
        :param y: the noisy measurement
        :param hermite_order: hermite order
        """
        self._y = y
        self._op = ObservationOperator(max_order_hermite=hermite_order)
        f_tmp = RandomGMM_DistributionFunction(modgrid=self._op.ssps).F
        self._f_shape = f_tmp.shape
        self._n_theta_v = hermite_order + 3

    def misfit(self, f, theta_v):
        f_img = np.reshape(f, self._f_shape)
        m = self._op.evaluate(f_img, theta_v) - self._y
        return m

    def misfitjac(self, f, theta_v):
        jac = self._evaluate_jacobian(f, theta_v)
        return jac

    def get_dims(self):
        """
        :return: _n_f_1, _n_f_2, _n_theta_v: the dimension of the parameters
        """
        return self._f_shape[0], self._f_shape[1], self._n_theta_v

    # private

    def _jac_f(self, f, theta_v):
        # compute dGdf
        d_tildeS_ft_df = self._op.ssps.F_tilde_s * self._op.ssps.delta_zt
        # turn it into twodim array
        # d_tildeS_ft_df_twodim = d_tildeS_ft_df.reshape(d_tildeS_ft_df.shape[0],-1)
        V, sigma, h, M = self._op.unpack_Theta_v(theta_v)
        losvd_ft = self._op.losvd.evaluate_fourier_transform(self._op.H_coeffs,
                                                             V,
                                                             sigma,
                                                             h,
                                                             M,
                                                             self._op.omega)
        d_ybar_ft_df = np.einsum('i,ijk->ijk', losvd_ft, d_tildeS_ft_df)
        d_ybar_ft_df_twodim = d_ybar_ft_df.reshape(d_ybar_ft_df.shape[0], -1)
        d_ybar_df = np.apply_along_axis(np.fft.irfft, 0, d_ybar_ft_df_twodim, self._op.ssps.n_fft)
        return d_ybar_df

    def _jac_theta_v(self, f, theta_v):
        f_img = np.reshape(f, self._f_shape)
        return self._op.partial_derivative_wrt_Theta_v(f_img, theta_v)

    def _evaluate_jacobian(self, f, theta_v):
        d_ybar_df = self._jac_f(f, theta_v)
        d_ybar_dthetav = self._jac_theta_v(f, theta_v)
        dybar = np.concatenate((d_ybar_df, d_ybar_dthetav), axis=1)
        return dybar