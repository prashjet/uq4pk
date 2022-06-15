import numpy as np


class ForwardOperator:

    m_f: int        # Number of metallicity bins.
    n_f: int        # Number of age bins.
    dim_theta: int  # Dimension of theta_v

    def fwd(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fwd_unmasked(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac_unmasked(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError