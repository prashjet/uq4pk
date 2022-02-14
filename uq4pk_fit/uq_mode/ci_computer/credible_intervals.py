

import numpy as np
from typing import Sequence


class CredibleInterval:
    """
    Output of the "compute_credible_intervals" function.
    """
    def __init__(self, phi_lower: np.ndarray, phi_upper: np.ndarray, minimizers: Sequence[np.ndarray],
                 maximizers: Sequence[np.ndarray]):
        self.phi_lower = phi_lower
        self.phi_upper = phi_upper
        self.minimizers = minimizers
        self.maximizers = maximizers