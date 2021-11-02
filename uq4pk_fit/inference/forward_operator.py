"""
Wrapper for the observation operator, including the possibility of a mask.
"""

import numpy as np

from uq4pk_src import model_grids
from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction

class ForwardOperator:

    def __init__(self, hermite_order=4, mask=None, ssps=model_grids.MilesSSP(), dv=10,
                 do_log_resample=True):
        """
        :param hermite_order: optional, int
            Order of the hermite expansion. Default is 4.
        :param mask: array_like, shape (M,)
            If the mask is 1, then the value is included in the measurement. If the mask is 0, then not.
        """
        self._op = ObservationOperator(max_order_hermite=hermite_order,
                                       ssps=ssps,
                                       dv=dv,
                                       do_log_resample=do_log_resample)
        f_tmp = RandomGMM_DistributionFunction(modgrid=self._op.ssps).F
        self.m_f = f_tmp.shape[0]
        self.n_f = f_tmp.shape[1]
        self.dim_f = self.m_f * self.n_f
        self.dim_theta = 3 + hermite_order
        self.grid = self._op.ssps.w
        self.modgrid = self._op.ssps
        # convert mask to indices and find measurement dimension
        y_tmp = self._op.evaluate(f_tmp, np.zeros(hermite_order + 3))
        if mask is None:
            self.dim_y = y_tmp.size
            self.mask = np.full((self.dim_y,), True, dtype=bool)
        else:
            self.mask = mask
            y_tmp_masked = y_tmp[mask]
            self.dim_y = y_tmp_masked.size

    def fwd(self, f, theta):
        """
        :param f: array_like, (N,)
        :param theta: (K,)
        :return: array_like, (M,)
        """
        f_im = np.reshape(f, (self.m_f, self.n_f))
        y = self._op.evaluate(f_im, theta)
        y_masked = y[self.mask]
        return y_masked

    def jac(self, f, theta):
        dy_df = self._jac_f(f, theta)
        dy_dtheta = self._jac_theta_v(f, theta)
        dydx = np.concatenate((dy_df, dy_dtheta), axis=1)
        dy_masked = dydx[self.mask, :]
        return dy_masked

    def downsample(self, f_im):
        return f_im

    def _jac_f(self, f, theta):
        """
        :param f: array_like, (N,)
        :param theta: array_like, (K,)
        :return: array_like, (M, N+K).
        """
        # compute dGdf
        d_tildeS_ft_df = self._op.ssps.F_tilde_s * self._op.ssps.delta_zt
        # turn it into twodim array
        # d_tildeS_ft_df_twodim = d_tildeS_ft_df.reshape(d_tildeS_ft_df.shape[0],-1)
        V, sigma, h, M = self._op.unpack_Theta_v(theta)
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

    def _jac_theta_v(self, f, theta):
        f_img = np.reshape(f, (self.m_f, self.n_f))
        return self._op.partial_derivative_wrt_Theta_v(f_img, theta)



