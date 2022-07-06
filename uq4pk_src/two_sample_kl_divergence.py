import numpy as np

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

    From https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


if __name__ == '__main__':
    from scipy import stats
    n = 10000
    d = 600
    mu_x = np.zeros(d)
    mu_z = np.ones(d)
    Sigma = np.eye(d)
    nrm_x = stats.multivariate_normal(mu_x, Sigma)
    nrm_z = stats.multivariate_normal(mu_z, Sigma)
    x1 = nrm_x.rvs(n)
    x2 = nrm_x.rvs(n)
    z = nrm_z.rvs(n)
    print(f'D_KL(x1,x2) = {KLdivergence(x1, x2)}')
    print(f'D_KL(x1,z) = {KLdivergence(x1, z)}')
    print(f'D_KL(x2,z) = {KLdivergence(x2, z)}')
    # output:
    # D_KL(x1,x2) = -0.42601092134229585
    # D_KL(x1,z) = 130.0265215059834
    # D_KL(x2,z) = 130.31610038714132




# end
