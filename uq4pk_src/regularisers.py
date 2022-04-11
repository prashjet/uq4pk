import numpy as np
from scipy import special


class Regulariser(object):
    """
    Quadratic regulariser \mathcal{R}(x) for least squares problem i.e.
        argmin_x { |Ax - b| + \mathcal{R}(x) }
    where
        \mathcal{R}(x ; Tik, x0, q, alpha) =
            (1-alpha) [ (x-x0)^T Q (x-x0) ] + alpha q^T x
            where Q = Tik^T Tik

    Parameters
    ----------
    tikhanov_matrix : array (ndim,ndim)
        sqrt of quadratic part of regulariser i.e. Tik in formula above
    alpha : float, in range 0-1
        fraction of regulariser allocated to the linear part
    linear_part : array (ndim,)
        linear part of the regulariser i.e. q in formula above
    x0 : array (ndim,)
        prior expected value of x

    Attributes
    ----------
    Rearrange the regulariser to the form:
        \mathcal{R}(x ; R, r) = x^T R x + r^T x + c
    R : (1-alpha) Q
    r : -2*(1-alpha) x0^T Q + alpha q^T
    c : (1-alpha) x0^T Q x0

    """

    def __init__(self,
                 tikhanov_matrix=None,
                 alpha=0.,
                 linear_part=None,
                 x0=None):
        assert (tikhanov_matrix is not None) or (linear_part is not None)
        if tikhanov_matrix is not None:
            self.ndim = tikhanov_matrix.shape[0]
        else:
            self.ndim = linear_part.shape[0]
            tikhanov_matrix = np.zeros((self.ndim, self.ndim))
        self.tikhanov_matrix = tikhanov_matrix
        assert (alpha>=0) and (alpha<=1)
        self.alpha = alpha
        if linear_part is None:
            linear_part = np.zeros(self.ndim)
        self.linear_part = linear_part
        if x0 is None:
            x0 = np.zeros(self.ndim)
        self.x0 = x0
        self.Q = np.dot(self.tikhanov_matrix.T, self.tikhanov_matrix)
        self.R = (1.-alpha) * self.Q
        self.r = -2.*(1.-alpha)*np.dot(self.Q, x0) + alpha*linear_part
        self.c = (1.-alpha)*np.einsum('i,ij,j', x0, self.Q, x0, optimize=True)

    def evaluate(self, x):
        result = np.einsum('...i,ij,...j', x, self.R, x, optimize=True)
        result += np.dot(x, self.r) + self.c
        if x.ndim == 1:
            result = result.item()
        return result


class L2(Regulariser):
    """
    \mathcal{R}(x) = sum_i x_i^2

    """

    def __init__(self, ndim=None):
        super().__init__(tikhanov_matrix=np.identity(ndim))


class L1(Regulariser):
    """
    \mathcal{R}(x) = sum_i |x_i|
                   = sum_i x_i      since we're only dealing with non-negative x

    """

    def __init__(self, ndim=None):
        super().__init__(alpha=1., linear_part=np.ones(ndim))


class ElasticNet(Regulariser):
    """
    \mathcal{R}(x) = (1-alpha) L2(x) + alpha L1(x)

    """

    def __init__(self, ndim=None, alpha=None):
        super().__init__(tikhanov_matrix=np.identity(ndim),
                         alpha=alpha,
                         linear_part=np.ones(ndim))


class GeneralisedTikhonov(Regulariser):
    """
    \mathcal{R}(x) = (x-x0)^T Q (x-x0)
        where Q = tikhanov_matrix^T tikhanov_matrix

    """

    def __init__(self, tikhanov_matrix=None, x0=None):
        super().__init__(tikhanov_matrix=tikhanov_matrix, x0=x0)


class GeneralisedL2(GeneralisedTikhonov):

    def __init__(self, ndim=None, x0=None):
        super().__init__(tikhanov_matrix=np.identity(ndim), x0=x0)


class FiniteDifference(GeneralisedTikhonov):
    """Generalised Tikhonov regularisation with finite difference operator as
    Tikhonov matrix.

    Parameters
    ----------
    modgrid : modgrid object
        model grid to which we want to apply finite difference regularisation
    axis : list
        list of grid axes over which we wish to regularise
    n : int/list
         order of regularisation i.e. we minimise the sum of squared n'th derivs
    ax_weights : list
         relative weight of regularisation in each axis
    x0 : array
        prior expected value of x.

    """

    def __init__(self,
                 modgrid=None,
                 axis=None,
                 n=0,
                 ax_weights=None,
                 x0=None):
        self.modgrid = modgrid
        if axis is None:
            axis = list(range(modgrid.npars))
        self.axis = axis
        n_ax = len(axis)
        if type(n) is int:
            n = n * np.ones(n_ax, dtype=int)
        else:
            n = np.array(n, dtype=int)
            assert len(n)==n_ax
        self.n = n
        if ax_weights is None:
            ax_weights = np.ones(n_ax)
        else:
            assert len(ax_weights)==n_ax
        self.ax_weights = ax_weights
        T = self.get_finite_difference_tikhonov_matrix()
        super().__init__(tikhanov_matrix=T, x0=x0)

    def get_finite_difference_tikhonov_matrix(self):
        # make C distance array
        # C[n, i, j] = signed distance in n'th parameter between pixel i and j
        shape = (self.modgrid.npars, self.modgrid.p, self.modgrid.p)
        C = np.full(shape, self.modgrid.p+1, dtype=int)
        for i in range(self.modgrid.npars):
            pi = self.modgrid.par_idx[i]
            Ci = pi - pi[:, np.newaxis]
            C[i, :, :] = Ci
        # make D distance arary
        # D[n, i, j] = { C[n, i, j] if other {1,...N}-{n} parameters are equal
        #              { self.p+1 otherwise (this won't interfere with T calc.)
        shape = (len(self.axis), self.modgrid.p, self.modgrid.p)
        D = np.full(shape, self.modgrid.p+1, dtype=int)
        all_axes = set(range(self.modgrid.npars))
        for i, ax0 in enumerate(self.axis):
            other_axes = all_axes - set([ax0])
            slice = np.sum(np.abs(C[tuple(other_axes), :, :]), 0)
            slice = np.where(slice==0)
            D[i][slice] = C[ax0][slice]
        # make tikhanov matrix
        shape = (len(self.axis), self.modgrid.p, self.modgrid.p)
        T = np.zeros(shape, dtype=int)
        for i, n0 in enumerate(self.n):
            k = np.arange(n0+1)
            d = int(np.ceil(n0/2.)) - k
            t = (-1)**k * special.binom(n0, k)
            for d0, t0 in zip(d, t):
                T[i][D[i]==d0] = t0
        # collapse along axes
        T = np.sum(T.T * self.ax_weights, -1).T
        return T
