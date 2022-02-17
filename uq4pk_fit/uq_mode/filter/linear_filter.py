
import numpy as np
import numpy.typing as npt


class LinearFilter:
    """
    A LINEAR filter is just a linear map vector -> value, defined by weights.
    """
    dim: int               # the dimension of the filter domain
    weights: np.ndarray    # Array containing the filter weights.

    def evaluate(self, v: np.ndarray) -> float:
        """
        Evaluates the filter. That is, it returns the value
            phi(v) = LinearFilter.weights @ v.
        :param v:
        :return: The value phi(v).
        """
        return self.weights @ v

    def extend(self, weights: np.ndarray):
        """
        Extends the filter by appending more indices with corresponding weights.
        :param weights: The corresponding weights.
        :raises ValueError: If inputs are inconsistent.
        """
        self.weights = np.concatenate([self.weights, weights])
        self.dim += weights.size
