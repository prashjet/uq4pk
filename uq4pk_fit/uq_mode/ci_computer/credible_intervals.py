

import numpy as np
from typing import Sequence


class CredibleInterval:
    """
    Output of the "compute_credible_intervals" function.
    """
    def __init__(self, phi_lower: np.ndarray, phi_upper: np.ndarray, time_avg: float = -1.):
        self.phi_lower = phi_lower.flatten()
        self.phi_upper = phi_upper.flatten()
        self.time_avg = time_avg