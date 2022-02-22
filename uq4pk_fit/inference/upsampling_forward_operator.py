
import numpy as np

from ..uq_mode import rectangle_partition

from uq4pk_src import model_grids
from uq4pk_fit.inference import ForwardOperator


class UpsamplingForwardOperator(ForwardOperator):

    def __init__(self, hermite_order=4, mask=None, ssps=model_grids.MilesSSP(), dv=10, do_log_resample=True,
                 scale: int = 1):
        """
        :param scale: Determines how much the scale is reduced, e.g. :code:`scale=2` corresponds to 2x2 superpixels.
        """
        ForwardOperator.__init__(self, hermite_order, mask, ssps, dv, do_log_resample)
        self._fine_op = ForwardOperator(hermite_order, mask, ssps, dv, do_log_resample)
        self.m_f_orig = self.m_f
        self.n_f_orig = self.n_f
        # make rectangle discretization of the image
        if scale == 1:
            a = 1
            b = 1
        elif scale == 2:
            a = 2
            b = 2
        elif scale == 3:
            a = 2
            b = 4
        else:
            raise RuntimeError("Unknown scale")
        self._partition = rectangle_partition(m=self.m_f, n=self.n_f, a=a, b=b)
        self.m_f = np.floor(self.m_f / a).astype(int)
        self.n_f = np.floor(self.n_f / b).astype(int)
        self.dim_f = self.m_f * self.n_f
        # check that I got this right
        assert self._partition.size == self.m_f * self.n_f
        # create upsampling jacobian
        self._upsampling_jacobian = self._create_upsampling_jacobian()

    def fwd(self, f_s, theta):
        """

        :param f_s: Of shape (floor(m_f/scale), floor(n_f/scale)).
        :param theta:
        :return:
        """
        # upsample f
        f = self._upsample(f_s)
        return self._fine_op.fwd(f, theta)

    def jac(self, f_s, theta):
        f = self._upsample(f_s)
        # compute Jacobian with respect to full distribution function
        dy_df = self._fine_op._jac_f(f, theta)
        # multiply with upsamling jacobian
        dy_df_s = dy_df @ self._upsampling_jacobian
        # rest is same as before
        dy_dtheta = self._fine_op._jac_theta_v(f, theta)
        dydx = np.concatenate((dy_df_s, dy_dtheta), axis=1)
        dy_masked = dydx[self.mask, :]
        return dy_masked

    def downsample(self, f_im):
        f = f_im.flatten()
        f_s_list = []
        for i in range(self._partition.size):
            mean_i = np.mean(f[self._partition.element(i)])
            f_s_list.append(mean_i)
        f_s = np.array(f_s_list)
        return f_s

    def _upsample(self, f_s: np.ndarray) -> np.ndarray:
        """
        Upsampling of coarser discretization
        """
        # use the Jacobian to make the upsampled image (as upsampling is linear)
        f = self._upsampling_jacobian @ f_s
        return f

    def _create_upsampling_jacobian(self) -> np.ndarray:
        """
        # The upsampling Jacobian is just a tall matrix with 1-columns of the size of the superpixels

        :return:
        """
        # Initialize:
        jac = np.zeros((self.m_f_orig * self.n_f_orig, self._partition.size))
        # Fill
        for j in range(self._partition.size):
            jac[self._partition.element(j), j] = 1.
        return jac