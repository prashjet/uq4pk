

from math import sqrt

from model_fit.models.curvelet_model.discrete_curvelet_transform import DiscreteCurveletTransform
from operators import RegularizationOperator

class CurveletCovroot(RegularizationOperator):
    """
    Corresponds to setting the Covroot equal to the inverse discrete curvelet transform.
    """

    def __init__(self, im_shape, nangles, nscales, weight):
        RegularizationOperator.__init__(self)
        self._dct = DiscreteCurveletTransform(n1=im_shape[0], n2=im_shape[1], nangles=nangles, nscales=nscales)
        self._weight = weight
        self.dim = im_shape[0] * im_shape[1]
        self.rdim = self._dct.ncoeff

    def fwd(self, v):
        """
        Evaluates the *forward* DCT.
        :param v:
        :return:
        """
        return self._dct.fwd(v) * sqrt(self._weight)

    def inv(self, w):
        """
        Computes the inverse DCT
        :param w: a vector
        :return:
        """
        return self._dct.inv(w) / sqrt(self._weight)

    def right(self, m):
        return m @ self._dct.PhiInv / sqrt(self._weight)

def curvelet_phi(im_shape, nangles, nscales, weight):
    dct = DiscreteCurveletTransform(n1=im_shape[0], n2=im_shape[1], nangles=nangles, nscales=nscales)
    return weight * dct.Phi