

from math import log
import numpy as np
import cvxpy as cp


def lci_optimization(alpha, fmap, jacmap, xmap1=None, xmap2=None, qinv=None, xbar1=None,
                            xbar2=None, s1inv=None, s2inv=None, options=None):
    """
    Computes projected credible intervals by convex optimization.
    :param alpha:
    :param xmap1:
    :param xmap2:
    :param fmap:
    :param jacmap:
    :param qinv:
    :param xbar1:
    :param xbar2:
    :param s1inv:
    :param s2inv:
    :param options:
    :return:
    """
    lci_optimizer = LciOptimizer(alpha, xmap1, xmap2, fmap, jacmap, qinv, xbar1, xbar2, s1inv, s2inv, options)
    lci_array = lci_optimizer.compute_lci(alpha)
    return lci_array


class LciOptimizer:

    def __init__(self, alpha, xmap1, xmap2, fmap, jacmap, qinv, xbar1, xbar2, s1inv, s2inv, options):
        # check input for consistency
        self._handle_input(alpha=alpha, xmap1=xmap1, xmap2=xmap2, fmap=fmap, jacmap=jacmap, qinv=qinv, xbar1=xbar1,
                                xbar2=xbar2, s1inv=s1inv, s2inv=s2inv, options=options)

    def compute_lci(self, alpha):
        """
        Computes the projected credible intervals via optimization
        :param alpha: A credibility parameter. A float satisfying 0 < alpha < 1.
        :return: An array of shape (n1+n2, 2), where the i-th row corresponds to the lower and upper bound of the i-th
        projected credible interval.
        """
        # compute tau_alpha
        tau = np.sqrt(16 * log(3. / alpha) / self._n)
        # compute eta
        eta = self._compute_eta(tau)
        return eta

    def _handle_input(self, alpha, xmap1, xmap2, fmap, jacmap, qinv, xbar1, xbar2, s1inv, s2inv, options):
        """
        Copy of the same method from pci_sampling -> Define superclass
        """
        # alpha must satisfy 0 < alpha < 1
        assert 0. < alpha < 1.
        if qinv is None:
            qinv = np.identity(fmap.size)
        else:
            assert qinv.shape[0] == qinv.shape[1] == fmap.size
        if options is None:
            options = {}
        n1 = 0
        n2 = 0
        if xmap1 is not None:
            n1 = xmap1.size
            if xbar1 is None:
                xbar1 = np.zeros(n1)
            else:
                assert xbar1.size == n1
            if s1inv is None:
                s1inv = np.identity(n1)
            else:
                assert s1inv.shape[1] == n1
            # if a1 is provided, its second dimension must match xmap1
            a1 = options.setdefault("a1", None)
            if a1 is not None:
                assert a1.shape[1] == xmap1.size
            # if b1 is provided, its shape must match a1
            b1 = options.setdefault("b1", None)
            if b1 is not None:
                assert b1.size == a1.shape[0]
        if xmap2 is not None:
            n2 = xmap2.size
            if xbar2 is None:
                xbar2 = np.zeros(n2)
            else:
                assert xbar2.size == n2
            if s2inv is None:
                s2inv = np.identity(n2)
            else:
                assert s2inv.shape[1] == n2
            # if a2 is provided, its second dimension must match xmap2
            a2 = options.setdefault("a2", None)
            if a2 is not None:
                assert a2.shape[1] == xmap2.size
            # if b1 is provided, its shape must match a1
            b2 = options.setdefault("b2", None)
            if b2 is not None:
                assert b2.size == a2.shape[0]
        # jacmap must be a matrix of shape (f_map.size, xmap1.size+xmap2.size)
        assert jacmap.shape == (fmap.size, n1 + n2)
        # set attributes
        self._fmap = fmap
        self._jacmap = jacmap
        self._n1 = n1
        self._n2 = n2
        self._n = n1 + n2
        self._xmap1 = xmap1
        self._xmap2 = xmap2
        if n1 > 0 and n2 > 0:
            self._xmap = np.concatenate((xmap1, xmap2))
        elif n1 > 0:
            self._xmap = xmap1
        elif n2 > 0:
            self._xmap = xmap2
        else:
            raise RuntimeError
        self._s1inv = s1inv
        self._s2inv = s2inv
        self._qinv = qinv
        self._xbar1 = xbar1
        self._xbar2 = xbar2
        self._a1 = options.setdefault("a1", None)
        self._b1 = options.setdefault("b1", None)
        self._a2 = options.setdefault("a2", None)
        self._b2 = options.setdefault("b2", None)

    def _psimap(self):
        """
        Computes psi(xmap)
        :return: The function psi as a cvxpy expression
        """
        if self._xmap1 is not None and self._xmap2 is not None:
            regterm = 0.5*np.linalg.norm(self._s1inv @ (self._xmap1 - self._xbar1), ord=1) \
                      + 0.5*np.linalg.norm(self._s1inv @ (self._xmap2-self._xbar2))**2
            misfit = 0.5*np.linalg.norm(self._qinv @ self._fmap)**2
        elif self._xmap1 is not None:
            regterm = 0.5*np.linalg.norm(self._s1inv @ (self._xmap1 - self._xbar1), ord=1)
            misfit = 0.5*np.linalg.norm(self._qinv @ self._fmap)**2
        elif self._xmap2 is not None:
            regterm = 0.5*np.linalg.norm(self._s2inv @ (self._xmap2 - self._xbar2))**2
            misfit = 0.5*np.linalg.norm(self._qinv @ self._fmap)**2
        else:
            raise RuntimeError
        psi = misfit + regterm
        return psi

    def _compute_eta(self, tau, partition):
        psimap = self._psimap()
        # define the constraints
        if self._n1 > 0 and self._n2 > 0:
            x1 = cp.Variable(self._n1)
            x2 = cp.Variable(self._n2)
            x = cp.vstack([x1, x2])
            regterm = 0.5*cp.norm(self._s1inv @ (x1 - self._xbar1), p=1) + 0.5*cp.sum_squares(self._s2inv @ (x - self._xbar2))
            jacmap1 = self._jacmap[:, :self._n1]
            jacmap2 = self._jacmap[:, self._n1]
            misfit = 0.5*cp.sum_squares(self._qinv @ (self._fmap + jacmap1 @ x1 + jacmap2 @ x2))
            psi = misfit + regterm
            constraints = [psi <= psimap + self._n * (tau + 1)]
            if self._a1 is not None:
                constraints += [self._a1 @ x1 >= self._b1]
            if self._a2 is not None:
                constraints += [self._a2 @ x >= self._b2]
        elif self._n1 > 0:
            x = cp.Variable(self._n1)
            regterm = 0.5*cp.norm(self._s1inv @ (x - self._xbar1), p=1)
            misfit = 0.5*cp.sum_squares(self._qinv @ (self._fmap + self._jacmap @ x))
            psi = misfit + regterm
            constraints = [psi <= psimap + self._n * (tau + 1)]
            if self._a1 is not None:
                constraints += [self._a1 @ x >= self._b1]
        elif self._n2 > 0:
            x = cp.Variable(self._n2)
            regterm = 0.5*cp.sum_squares(self._s2inv @ (x - self._xbar2))
            misfit = 0.5*cp.sum_squares(self._qinv @ (self._fmap + self._jacmap @ x))
            psi = misfit + regterm
            tresh = self._n * (tau + 1)
            constraints = [psi <= psimap + tresh]
            if self._a2 is not None:
                constraints += [self._a2 @ x >= self._b2]
        else:
            raise RuntimeError
        eta = np.zeros((self._n, 2))
        # warm start
        x.value = self._xmap
        for i in range(self._n):
            print(f"Computing {i}-th projected credible interval.")
            zeta = np.zeros(self._n)
            zeta[i] = 1.
            prob_i_min = cp.Problem(cp.Minimize(zeta @ x), constraints)
            prob_i_max = cp.Problem(cp.Maximize(zeta @ x), constraints)
            prob_i_min.solve(verbose=True, feastol=1e-1, abstol=1e-1, warm_start=True)
            print("Minimization done.")
            eta[i, 0] = prob_i_min.value
            print(f"eta[{i},0]={eta[i,0]}")
            prob_i_max.solve(verbose=True, feastol=1e-1, abstol=1e-1, warm_start=True)
            eta[i, 1] = prob_i_max.value
            print(f"eta[{i},1]={eta[i, 1]}")
        return eta

    def _combine_constraints(self):
        # this function is probably unnecessary
        """
        Combines the constraints a1 @ x1 >= b1 and a2 @ x2 >= b2 into a common constraint
        a @ x >= b.
        :param a1:
        :param a2:
        :param b1:
        :param b2:
        :return: a, b, where either both a and b are None, indicating no constraints,
        or a is a matrix of shape (a1.shape[0]+a2.shape[0], self._n) and b is a vector of size (b1.size+b2.size),
        (with the convention None.shape[0] = 0 and None.size = 0)
        """
        if self._n1 == 0:
            # only have l2 problem
            a = self._a2
            b = self._b2
        elif self._n2 == 0:
            # only have l2 problem
            a = self._a1
            b = self._b1
        elif self._n1 > 0 and self._n2 > 0:
            if self._a1 is not None and self._a2 is not None:
                # no constraints
                a = None
                b = None
            else:
                if self._a1 is None:
                    # default the constraints
                    a1 = np.zeros((0, self._n1))
                    b1 = np.zeros(1)
                else:
                    a1 = self._a1
                    b1 = self._b1
                if self._a2 is None:
                    # default constraints
                    a2 = np.zeros((0, self._n1))
                    b2 = np.zeros(1)
                else:
                    a2 = self._a2
                    b2 = self._b2
                # combine the constraints
                a = np.concatenate((a1, a2), axis=1)
                b = np.concatenate((b1, b2))
        else:
            raise RuntimeError
        return a, b