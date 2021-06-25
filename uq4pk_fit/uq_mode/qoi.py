
from math import sqrt, log
import numpy as np

from .convex_optimizer import ConvexOptimizer


class QuantityOfInterest:

    def __init__(self, alpha, costfun, costgrad, xmap, lb):
        self._n = xmap.size
        self._tau = sqrt(16 * log(3/alpha) / self._n)
        self._xmap = xmap
        self._mapcost = costfun(xmap)
        self.costfun = costfun
        self.costgrad = costgrad
        self.lb = lb
        self.optimizer = ConvexOptimizer()
        self.tol = 0.1

    def constraint(self, z):
        c = self._mapcost + self._n * (self._tau + 1) - self.costfun(self.x(z))
        return c

    def constraint_jac(self, z):
        jac = - self.costgrad(self.x(z)) @ self.dx_dz(z)
        return jac

    def lossfun(self, z):
        raise NotImplementedError

    def lossgrad(self, z):
        raise NotImplementedError

    def x(self, z):
        raise NotImplementedError

    def dx_dz(self, z):
        raise NotImplementedError

    def initial_value(self):
        raise NotImplementedError

    def lower_bound(self):
        raise NotImplementedError

    def compute(self):
        """
        Computes the quantity of interest.
        """
        objective = {"fun": self.lossfun, "grad": self.lossgrad}
        constraint = {"fun": self.constraint, "jac": self.constraint_jac}
        z0 = self.initial_value()
        lb = self.lower_bound()
        z = self.optimizer.optimize(objective=objective, constraint=constraint, lb=lb, x0=z0)
        constraint_error = self.negative(self.constraint(z))
        if constraint_error > self.tol:
            z = self.optimizer.optimize(objective=objective, constraint=constraint, lb=lb, x0=z0,
                                        conservative=True)
        return z

    def negative(self, v):
        vminus = -v
        vneg = vminus.clip(min=0.)
        return vneg


class LCI(QuantityOfInterest):

    def __init__(self, alpha, costfun, costgrad, xmap, lb, ind, type):
        QuantityOfInterest.__init__(self, alpha, costfun, costgrad, xmap, lb)
        if type=="lower":
            self._lower = True
        elif type=="upper":
            self._lower = False
        else:
            raise Exception("Unknown type.")
        self._xmap = xmap
        self._ind = ind
        self._zeta = np.zeros(self._n)
        self._zeta[ind] = 1.


    def lossfun(self, xi):
        if self._lower:
            return xi
        else:
            return -xi

    def lossgrad(self, z):
        if self._lower:
            return 1.
        else:
            return -1.

    def x(self, xi):
        x_xi = self._xmap.copy()
        x_xi[self._ind] += xi
        return x_xi

    def dx_dz(self, xi):
        return self._zeta

    def initial_value(self):
        return np.reshape(self._xmap[self._ind[0]], (1,))

    def lower_bound(self):
        return np.array([np.max(self.lb[self._ind])])


class Maxleft(QuantityOfInterest):

    def __init__(self, alpha, costfun, costgrad, xmap, lb, ind, i):
        QuantityOfInterest.__init__(self, alpha, costfun, costgrad, xmap, lb)
        self._xmap = xmap
        self._ind = ind
        self._j = ind.size
        self._i = i
        self.x_jac = np.zeros((self._n, self._j))
        id_j = np.identity(self._j)
        self.x_jac[self._ind, :] = id_j[:, :]
        self._lossgrad = np.zeros(self._j)
        self._lossgrad[self._i] = 1.

    def lossfun(self, d):
        lf = d[self._i]
        return lf

    def lossgrad(self, d):
        lg = self._lossgrad
        return lg

    def x(self, d):
        x_d = self._xmap.copy()
        x_d[self._ind] += d
        return x_d

    def dx_dz(self, z):
        return self.x_jac

    def initial_value(self):
        d0 = np.zeros(self._ind.size)
        return d0

    def lower_bound(self):
        return self.lb[self._ind]

class Maxright(QuantityOfInterest):

    def __init__(self, alpha, costfun, costgrad, xmap, lb, ind, i):
        QuantityOfInterest.__init__(self, alpha, costfun, costgrad, xmap, lb)
        self._xmap = xmap
        self._ind = ind
        self._j = ind.size
        self._i = i
        self.x_jac = np.zeros((self._n, self._j))
        id_j = np.identity(self._j)
        self.x_jac[self._ind, :] = id_j[:, :]
        self._lossgrad = np.zeros(self._j)
        self._lossgrad[self._i] = 1.

    def lossfun(self, d):
        lf = -d[self._i]
        return lf

    def lossgrad(self, d):
        lg = -self._lossgrad
        return lg

    def x(self, d):
        x_d = self._xmap.copy()
        x_d[self._ind] += d
        return x_d

    def dx_dz(self, d):
        return self.x_jac

    def initial_value(self):
        d0 = np.zeros(self._ind.size)
        return d0

    def lower_bound(self):
        return self.lb[self._ind] - self._xmap[self._ind]



class MaxMinReg(QuantityOfInterest):

    def __init__(self, alpha, costfun, costgrad, xmap, lb, regterm, reggrad, type):
        QuantityOfInterest.__init__(self, alpha, costfun, costgrad, xmap, lb)
        if type == "max":
            self._max = True
        elif type == "min":
            self._max = False
        else:
            raise Exception("Unknown type.")
        self._xmap = xmap
        self._regterm = regterm
        self._reggrad = reggrad

    def lossfun(self, z):
        if self._max:
            return self._regterm(z)
        else:
            return - self._regterm(z)

    def lossgrad(self, z):
        if self._max:
            return self._reggrad(z)
        else:
            return - self._reggrad(z)

    def x(self, z):
        return z

    def dx_dz(self, z):
        return np.identity(self._n)

    def initial_value(self):
        return self._xmap

    def lower_bound(self):
        return self.lb
