import numpy as np
from scipy import stats
from . import distribution_function, losvds

class Noise:

    def __init__(self,
                 n,
                 sig=None,
                 sig_i=None,
                 cov=None):
        # check input
        self.n = n
        c1 = sig is not None
        c2 = sig_i is not None
        c3 = cov is not None
        if c1 + c2 + c3 != 1:
            errmsg = 'Exactly one of sig, sig_i or cov must be set'
            raise ValueError(errmsg)
        if c1:
            self.sig = sig
            self.sig_i = np.ones(self.n)*sig
            self.cov = sig**2 * np.identity(n)
            self.pre = sig**-2 * np.identity(n)
        if c2:
            assert sig_i.shape == (n,)
            self.sig_i = sig_i
            self.cov = np.diag(sig_i**2)
            self.pre = np.diag(sig_i**-2)
        if c3:
            assert cov.shape == (n,n)
            self.cov = cov
            self.pre = np.linalg.inv(cov)

    def sample(self):
        if hasattr(self, 'sig_i'):
            nrm = stats.norm()
            noise = nrm.rvs(size=self.n)*self.sig_i
        else:
            mvn = stats.multivariate_normal(mean=np.zeros(self.n),
                                            cov=self.cov)
            noise = mvn.rvs()
        return noise

    def transform_subscripts(self, subscripts):
        # split subscripts
        if '->' in subscripts:
            ss_in, ss_out = subscripts.split('->')
            has_output = True
        else:
            ss_in = subscripts
            has_output = False
        ss_x, ss_cov, ss_y = subscripts.split(',')

        # transform subscripts
        if hasattr(self, 'sig'):
            ss_x = ss_x.replace(ss_cov[0], ss_cov[1])
            ss = f'{ss_x},{ss_y}'
        # isotropic case
        elif hasattr(self, 'sig_i'):
            ss_x = ss_x.replace(ss_cov[0], ss_cov[1])
            ss = f'{ss_x},{ss_cov[1]},{ss_y}'
        # general case
        else:
            ss = ss_in
        if has_output is True:
            ss + '->' + ss_out
        return ss

    def einsum_x_cov_y(self, subscripts, x, y, precision=False):
        '''
        Fast evaluation of products with covariance or precision matrix.
        Optimised for cases of istotropic/diagonal/general noise.
        Evaluates
            np.einsum(subscripts, x, cov, y) if precision=False
            np.einsum(subscripts, x, pre, y) if precision=True
        Subscripts must be a valid einsum string of the type
            '{sx}i,ij,j{sy}'
        possibly appended with a valid output string of the type
            '->{s_out}'
        '''
        ss = self.transform_subscripts(subscripts)
        if hasattr(self, 'sig'):
            if precision:
                xcy = self.sig**-2. * np.einsum(ss, x, y, optimize='true')
            else:
                xcy = self.sig**2. * np.einsum(ss, x, y, optimize='true')
        # isotropic case
        elif hasattr(self, 'sig_i'):
            if precision:
                xcy = np.einsum(ss, x, self.sig_i**-2., y, optimize='true')
            else:
                xcy = np.einsum(ss, x, self.sig_i**2., y, optimize='true')
        # general case
        else:
            if precision:
                xcy = np.einsum(ss, x, self.pre, y, optimize='true')
            else:
                xcy = np.einsum(ss, x, self.cov, y, optimize='true')
        return xcy


class Data:

    def __init__(self,
                 lmd=None,
                 y=None):
        n = lmd.size
        if lmd.size!=y.size:
            errmsg = 'lmd and y must have same length'
            raise ValueError(errmsg)
        self.n = n
        self.lmd = lmd
        self.y = y


class MockData(Data):

    def __init__(self,
                 ssps=None,
                 df=None,
                 losvd=None,
                 snr=None):
        n, p = ssps.X.shape
        self.ssps = ssps
        self.df = df
        self.S = np.dot(ssps.X, df.beta)
        self.losvd = losvd
        self.ybar = losvd.convolve(S=self.S, lmd_in=ssps.lmd)
        self.snr = snr
        sig = np.mean(np.abs(self.ybar))/snr
        noise = Noise(n, sig=sig)
        y = self.ybar + noise.sample()
        super().__init__(lmd=ssps.lmd,
                         y=y)


class RandomMockData(MockData):

    def __init__(self,
                 ssps=None,
                 snr=None):
        # generate random DF and losvd
        df = distribution_function.DistributionFunction(ssps)
        losvd = losvds.InputLOSVD()
        super().__init__(ssps=ssps,
                         df=df,
                         losvd=losvd,
                         snr=snr)

# end
