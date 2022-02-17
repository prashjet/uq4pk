
import numpy as np


class FilterKernel:
    """
    A filter kernel is just a function.
    """
    dim: int        # The dimension of the continuous signal (e.g. 2 for images)

    def weighting(self, v: np.ndarray) -> float:
        """
        Evaluates the weighting function in vectorized form.
        """
        raise NotImplementedError