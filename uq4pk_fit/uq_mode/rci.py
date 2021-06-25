"""
This test is mainly about determining whether to use LCI, SLCI or minmaxreg.
A second test will use the winner-method on different regularization-operators/setups.
"""


import numpy as np

from .credible_interval_optimizer import QoiComputer
from .qoi import Maxright

def rci(alpha, partition, n, xmap, cost, lb=None):
    """
    Computes the semilocal posterior credible interval for the statistical model
    Y = H @ X + a + V, X ~ normal(xmean, s @ s.T), V ~ normal(0, delta^2 * identity)
    :return: ndarray of shape (len(partition), 2)
        The j-th row contains the lower and upper bound for the semilocal credible interval corresponding to
        the j-th element of 'partition'
    """
    # default lb
    if lb is None:
        lb = -np.inf * np.ones(n)
    # initialize an lci-object and feed it the problem object
    rci_computer = RciComputer(alpha=alpha, partition=partition, n=n, xmap=xmap, cost=cost, lb=lb)
    # compute the slcis
    rci_array = rci_computer.compute()
    # return the slcis
    return rci_array


class RciComputer(QoiComputer):

    def __init__(self, alpha, n, xmap, cost, lb, partition):
        QoiComputer.__init__(self, alpha, n, xmap, cost, lb)
        self._partition = partition
        self._J = len(partition)

    def compute(self):
        """
        Computes semi-local credible intervals. For each partition element i, returns
        [-x_map - eta, x_map + eta],
        where eta = argmax(||x[ind]-x_map[ind]||^2: x in credible region).
        That is, this function solves for every partition element an optimization problem of size ind.size.
        :return: ndarray of shape (n, 2)
        """
        # initialize array of shape (n, 2)
        rci_array = np.zeros((self._n, 2))
        # for ind_j in partition
        for j in range(self._J):
            print(f"Computing RCI {j+1}/{self._J}")
            ind = self._partition[j]
            rci_array[ind, :] = self._compute_rci(ind)[:, :]
        return rci_array

    def _compute_rci(self, ind):
        """
        For a given index set, computes the corresponding semilocal credible interval [xi_min, xi_max], where
        xi_min = xmap - u, xi_max = xmap + u, and
        u = max {||x(z) - xmap||^2 : phi(x(z,ind)) <= phi(xmap) + n * (tau+1), x(z,ind) >= lb }.
        where x(z,ind)[i] = z[i], for i in ind, and x(z,ind)[i] = xmap[i] otherwise.
        :param ind: ndarray of type int.
        :return: ndarray of shape (ind.size, 2)
        """
        d_plus = np.zeros(ind.size)
        for i in range(ind.size):
            right_deviation = Maxright(alpha=self._alpha, costfun=self._costfun, costgrad=self._costgrad, xmap=self._xmap,
                                       lb=self._lb, ind=ind, i=i)
            d_plus[i] = right_deviation.compute()[i]
        xmin = (self._xmap[ind] - d_plus).clip(min=self._lb[ind])
        xmax = self._xmap[ind] + d_plus
        return np.column_stack((xmin, xmax))