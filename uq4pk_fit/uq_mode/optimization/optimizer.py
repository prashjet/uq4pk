
import numpy as np
from typing import Literal

from .socp import SOCP


class Optimizer:

    def setup_problem(self, socp: SOCP, ctol: float, mode: Literal["min", "max"]):
        raise NotImplementedError

    def optimize(self) -> np.ndarray:
        """
        Solves the SOCP and returns the optimizer.
        """
        raise NotImplementedError

    def change_loss(self, w: np.ndarray):
        raise NotImplementedError
