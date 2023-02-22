
import numpy as np

from ..operators import DiagonalOperator, OrnsteinUhlenbeck
from ..optimization import NonnegativeLinearModel
from .forward_operator import ForwardOperator


class StatModel:
    """
    Abstract base class for that manages the optimization problem, regularization, and optionally also the
    uncertainty quantification.
    The full statistical model is
    y ~ fwd(f) + error,
    error ~ normal(0, standard_deviation**2 * identity),
    f ~ normal(f_bar, cov1), where cov1 = (beta * P @ P.T)^(-1),
    f >= 0.

    Attributes
    -------
    beta
        The regularization parameter. Defaults to `1000 * snr`.
    P
        The regularization operator. Defaults to `OrnsteinUhlenbeck(m=self.m_f, n=self.n_f, h=h)`.
    """

    def __init__(self, y: np.ndarray, y_sd: np.ndarray, forward_operator: ForwardOperator):
        """
        Parameters
        ----------
        y : shape (n, )
            The masked data vector.
        y_sd : shape (n, )
            The masked vector of standard deviations
        forward_operator
            The operator that maps stellar distribution functions to corresponding observations.
            Must satisfy `forward_operator.dim_y = n`.
        """
        # Check that the dimensions match.
        m = y.size
        assert y_sd.size == m
        assert forward_operator.dim_y == m
        self._op = forward_operator
        y_sum = np.sum(y)
        y_scaled = y / y_sum
        y_sd_scaled = y_sd / y_sum
        self._y = y_scaled
        self._scaling_factor = y_sum
        self._sigma_y = y_sd_scaled
        self._R = DiagonalOperator(dim=y.size, s=1 / y_sd_scaled)
        # get parameter dimensions from misfit handler
        self._m_f = forward_operator.m_f
        self._n_f = forward_operator.n_f
        self._dim_f = self._m_f * self._n_f
        self._dim_y = y.size
        self._snr = np.linalg.norm(y_scaled) / np.linalg.norm(y_sd_scaled)
        # SET DEFAULT PARAMETERS
        self._lb_f = np.zeros(self._dim_f)
        self._scale = self._dim_y
        # SET DEFAULT REGULARIZATION PARAMETERS
        self.beta = 1e3 * self._snr
        self.f_bar = np.zeros(self._dim_f) / self._scaling_factor
        h = np.array([4., 2.])
        self.P = OrnsteinUhlenbeck(m=self._m_f, n=self._n_f, h=h)
        # Initialize solver

    @property
    def y(self) -> np.ndarray:
        """
        The RESCALED masked data vector.
        """
        return self._y

    @property
    def sigma_y(self) -> np.ndarray:
        """
        The RESCALED masked vector of standard deviations.
        """
        return self._sigma_y

    @property
    def m_f(self) -> int:
        """
        Number of rows for the image of a distribution function.
        """
        return self._m_f

    @property
    def n_f(self) -> int:
        """
        Number of columns for the image of a distribution function.
        """
        return self._n_f

    @property
    def dim_f(self) -> int:
        """
        Dimension (=no. of pixels) of the distribution function. This is just `m_f * n_f`.
        """
        return self._dim_f

    @property
    def dim_y(self) -> int:
        """
        Dimension of the masked data vector. This is just `_y.size`.
        """
        return self._dim_y

    @property
    def snr(self) -> float:
        """
        The signal-to-noise ratio `||_y||/||sigma_y||`.
        """
        return self._snr

    def compute_map(self) -> np.ndarray:
        """
        Computes the MAP estimator for the model.

        Returns
        ---
        shape (m_f, n_f)
            The MAP estimate (in image format).
        """
        model = NonnegativeLinearModel(y=self.y, P_error=self._R, G=self._op.mat, P_f=self.P, beta=self.beta,
                                       f_bar=self.f_bar, scaling_factor=self._scale)
        f_map = self._scaling_factor * model.fit()
        f_map_image = f_map.reshape(self.m_f, self.n_f)
        return f_map_image
