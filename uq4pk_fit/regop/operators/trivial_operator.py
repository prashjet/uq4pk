
import numpy as np

from ..regularization_operator import RegularizationOperator



class TrivialOperator(RegularizationOperator):
    """
    Corresponds to to the identity operator
    """
    def __init__(self, dim):
        RegularizationOperator.__init__(self)
        self.dim = dim
        self.rdim = dim
        self.mat = np.identity(dim)
        self.imat = np.identity(dim)

    def fwd(self, v):
        return v

    def inv(self, v):
        return v

    def right(self, v):
        return v