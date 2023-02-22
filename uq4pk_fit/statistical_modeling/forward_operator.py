import numpy as np


class ForwardOperator:
    """
    Abstract base class of a forward operator, i.e. something that maps stellar distribution functions to spectra.
    """
    m_f: int  # Number of metallicity bins.
    n_f: int  # Number of age bins.
    dim_theta: int  # Dimension of theta_v
    dim_y: int  # Number of data points (with mask applied).
    dim_y_unmasked: int  # Number of data points (without mask). Should be the same as `mask.size`.

    def fwd(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def mat(self) -> np.ndarray:
        raise NotImplementedError

    def fwd_unmasked(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def mat_unmasked(self) -> np.ndarray:
        raise NotImplementedError