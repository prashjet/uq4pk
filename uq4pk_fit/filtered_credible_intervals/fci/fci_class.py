
import numpy as np


class FCI:
    """
    Object that is returned by the function "fci".
    """
    def __init__(self, lower_stack: np.ndarray, upper_stack: np.ndarray, mean: np.ndarray = None, time_avg: float = -1):
        assert upper_stack.shape == lower_stack.shape
        self.num_scales = lower_stack.shape[0]
        self.num_pixels = lower_stack.shape[1]
        self.mean = mean
        if lower_stack.shape[0] == 1:
            self.lower = lower_stack.flatten()
            self.upper = upper_stack.flatten()
        else:
            self.lower = lower_stack
            self.upper = upper_stack
        self.time_avg = time_avg