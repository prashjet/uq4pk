"""
Contains the function 'lci'.
"""

import numpy as np

from .credible_interval_optimizer import QoiComputer
from .qoi import LCI


def lci(alpha, partition, n, xmap, cost, lb=None):
    """
    Returns the (1-alpha)-locally credible intervals around 'xmap' corresponding to 'partition' for the statistical
    model:
    y = h @ x + a + v
    v ~ normal(0, delta^2 * id)
    x ~ normal(xbar, s @ s.T)
    x >= l
    :param alpha:
    :param h:
    :param y:
    :param a:
    :param delta:
    :param xbar:
    :param s:
    :param lb:
    :param partition:
    :return: ndarray of shape (n,2)
    """
    # default lb
    if lb is None:
        lb = -np.inf * np.ones(n)
    # initialize an lci-object and feed it the problem object
    lci_computer = LciComputer(alpha=alpha, partition=partition, n=n, xmap=xmap, cost=cost, lb=lb)
    # compute the lcis
    lci_array = lci_computer.compute()
    # return the lcis
    return lci_array


class LciComputer(QoiComputer):

    def __init__(self, alpha, n, xmap, cost, lb, partition):
        QoiComputer.__init__(self, alpha, n, xmap, cost, lb)
        self._partition = partition
        self._J = len(partition)

    def compute(self):
        """
        Computes the local credible intervals
        """
        # initialize xi as array of shape (n,2)
        xi = np.zeros((self._n, 2))
        # for ind_j in partition
        for j in range(self._J):
            print(f"Computing LCI {j+1}/{self._J}")
            ind = self._partition[j]
            xi[ind, :] = self._compute_xi(ind)
        return xi

    def _compute_xi(self, ind):
        """
        Computes [xi_min, xi_max] where
        xi_min = argmin { xi : phi(x(xi, ind) <= phi(xmap) + n*(tau+1), lb <= x(xi) }
        xi_max = argmax {'''}
        :param ind: ndarray of type int
            The index set that determines the partition for which the locally credible interval is computed.
        :return: ndarray of shape (ind.size, 2)
            Returns the lower and upper bound of the locally credible interval: [xmap + xi_min, xmap + xi_max]
        """
        # setup optimization problem in scipy
        Xi_min = LCI(alpha=self._alpha, costfun=self._costfun, costgrad=self._costgrad, xmap=self._xmap, lb=self._lb,
                     ind=ind, type="lower")
        Xi_max = LCI(alpha=self._alpha, costfun=self._costfun, costgrad=self._costgrad, xmap=self._xmap, lb=self._lb,
                     ind=ind, type="upper")
        xi_min = Xi_min.compute()[0]
        xi_max = Xi_max.compute()[0]
        print(f"xi_min={xi_min}")
        print(f"xi_max={xi_max}")
        lower = self._xmap[ind] + xi_min
        upper = self._xmap[ind] + xi_max
        return np.column_stack((lower, upper))