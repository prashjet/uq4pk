
from copy import deepcopy
import numpy as np
from typing import Union

from ...cgn import RegularizationOperator

class LinearModel:
    """
    Container for a constrained linear model of the form
        y = H @ x + eta
        eta ~ normal(0, Gamma), Gamma = (Q Q^\top)^(-1)
        x ~ normal(m, Sigma), Sigma = (R R^\top)^(-1)
        A x = b
        x >= lb
    """
    def __init__(self, h: np.ndarray, y: np.ndarray, q: RegularizationOperator, m: np.ndarray, r: RegularizationOperator,
                 a: Union[np.ndarray, None], b: Union[np.ndarray, None], lb: Union[np.ndarray, None]):
        # Check input for consistency
        # Copy input in instance attributes
        self.h = deepcopy(h)
        self.ydim = int(y.size)
        self.y = deepcopy(y)
        self.q = deepcopy(q)
        self.m = deepcopy(m)
        self.r = deepcopy(r)
        self.a = deepcopy(a)
        self.b = deepcopy(b)
        if lb is None:
            self.lb = - np.inf * np.ones((m.size, ))
        else:
            self.lb = deepcopy(lb)
        # store further information to save computation time
        self.n = int(m.size)
        self.qh = self.q.fwd(self.h)

    def cost(self, x: np.ndarray) -> float:
        """
        Returns the cost function at x:
        phi(x) = 0.5 * ||Q(H x - y)||_2^2 + 0.5 * ||R(x - m)||_2^2
        :param x:
        :return:
        """
        misfit = np.sum(np.square(self.q.fwd(self.h @ x - self.y)))
        regularization = np.sum(np.square(self.r.fwd(x - self.m)))
        return 0.5 * misfit + 0.5 * regularization

    def cost_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the cost function
        :param x:
        :return:
        """
        misfit_grad = self.qh.T @ self.q.fwd(self.h @ x - self.y)
        regularization_grad = self.r.adj(self.r.fwd(x - self.m))
        return misfit_grad + regularization_grad