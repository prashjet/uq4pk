
import numpy as np


class LinearFilter:
    """
    A LINEAR filter is just a linear map vector -> value, defined by weights.
    """
    def __init__(self, weights: np.ndarray):
        assert weights.ndim == 1
        self.weights = weights
        self.dim = weights.size

    def evaluate(self, v: np.ndarray) -> float:
        return self.weights @ v

    def extend(self, weights: np.ndarray, before: bool = False):
        if before:
            self.weights = np.concatenate([weights, self.weights])
        else:
            self.weights = np.concatenate([self.weights, weights])
        self.dim += weights.size
        assert self.dim == self.weights.size
