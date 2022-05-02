
import numpy as np
from typing import Literal

from .socp import SOCP


class Optimizer:
    def optimize(self, problem: SOCP, start: np.ndarray, ctol: float, mode: Literal["min", "max"]) -> np.ndarray:
        raise NotImplementedError
