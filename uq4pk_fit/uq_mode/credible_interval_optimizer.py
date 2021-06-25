
import numpy as np

from uq4pk_fit.regop import RegularizationOperator


class NnlsProblem():

    def __init__(self, h, y, a, delta, xmean, s: RegularizationOperator, lb):
        # read input
        self._read_input(h, y, a, delta, xmean, s, lb)
        # check consistency
        self._check_consistency()
        pass

    def _read_input(self, h, y, a, delta, xmean, s, lb):
        self.h = h
        self.y = y
        self.a = a
        self.delta = delta
        self.xmean = xmean
        self.s = s
        if lb is None:
            self.lb = np.zeros(xmean.size)
            self.lb[:] = - np.inf
        else:
            self.lb = lb

    def _check_consistency(self):
        # y and h must match
        assert self.y.size == self.h.shape[0] == self.a.size
        # h and xmean must match
        assert self.xmean.size == self.h.shape[1]
        # xmean and lb must match
        assert self.xmean.size == self.lb.size
        # delta > 0
        assert self.delta > 0.



class QoiComputer():
    """
    Superclass for computation of credible intervals
    """

    def __init__(self, alpha, n, xmap, cost, lb):
        # precompute:
        self._alpha = alpha
        self._n = n
        self._xmap = xmap
        self._costfun = cost["fun"]
        self._costgrad = cost["grad"]
        self._lb = lb

    def compute(self):
        raise NotImplementedError

    def _sumsquares(self, v):
        return np.inner(v, v)