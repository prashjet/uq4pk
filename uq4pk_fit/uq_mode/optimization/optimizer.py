
import numpy as np

from .socp import SOCP


class Optimizer:
    def optimize(self, problem: SOCP, start: np.ndarray) -> np.ndarray:
        raise NotImplementedError