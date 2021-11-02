
from copy import deepcopy
import numpy as np


class Constraint:

    def fun(self, x):
        raise NotImplementedError

    def jac(self, x):
        raise NotImplementedError

    def as_dict(self):
        raise NotImplementedError


class NonlinearConstraint(Constraint):
    def __init__(self, fun, jac, type):
        self._cfun = deepcopy(fun)
        self._cjac = deepcopy(jac)
        self._type = type

    def fun(self, x):
        return self._cfun(x)

    def jac(self, x):
        return self._cjac(x)

    def as_dict(self):
        dict = {"type": self._type, "fun": self.fun, "jac": self.jac}
        return dict


class NullConstraint(Constraint):
    def fun(self, x):
        return 0.

    def jac(self, x):
        return np.zeros((1, x.size))

    def as_dict(self):
        return None
