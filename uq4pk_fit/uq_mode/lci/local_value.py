"""
Contains class "LocalValue".
"""

import numpy as np
from typing import Union

from ..optimization import Constraint, NonlinearConstraint


class LocalValue:
    """
    Represent the local credible intervals as proposed by Cai et al. (2017).
    """
    def __init__(self, x_map: np.ndarray, indices: np.ndarray):
        self._xmap = x_map
        n = x_map.size
        self._ind = indices
        self._zeta = np.zeros(n)
        self._zeta[indices] = 1.

    def loss_fun(self, xi: float) -> float:
        return xi

    def loss_grad(self, xi: float) -> float:
        return 1.

    def transform_nonlinear_constraint(self, fun: callable, jac: callable, type: str) -> Constraint:
        """
        Transforms a constraint of the form fun(x) >= 0 (or jac(x) >= 0) to z-space.
        """
        def fun_z(z):
            x = self.x(z)
            y = fun(x)
            return y
        def jac_z(z):
            x = self.x(z)
            j = jac(x) @ self.dx_dz(z)
            return j
        constraint_z = NonlinearConstraint(fun=fun_z, jac=jac_z, type=type)
        return constraint_z

    def x(self, xi: float) -> np.ndarray:
        """
        Sets x equal to xi inside 'ind' and equal to 'xmap' outside.
        :param xi: float
        :return: (n,) array
        """
        x = self._xmap.copy()
        x[self._ind] += xi
        return x

    def dx_dz(self, xi: float) -> np.ndarray:
        return self._zeta

    @property
    def initial_value(self) -> np.ndarray:
        return np.zeros((1,))

    def lower_bound(self, lb: Union[np.ndarray, None]) -> Union[None, np.ndarray]:
        if lb is None:
            return None
        else:
            return np.array([np.max(lb[self._ind] - self._xmap[self._ind])])