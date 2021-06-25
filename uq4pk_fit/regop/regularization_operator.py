"""
Contains class 'RegularizationOperator'.
"""


class RegularizationOperator:
    """
    Abstract base class for regularization operators.
    Every regularization operator P is an injective linear map from R^n to R^r,
    with a left-inverse which maps from R^r to R^n.
    Using P as regularization operator corresponds to a prior covariance proportional to P @ P.T
    """
    def __init__(self):
        self.mat = None
        self.imat = None
        self.dim = 0
        self.rdim = 0

    def right(self, v):
        """
        Computes the right multiplication with the INVERSE regularization operator.
        :param v: ndarray of shape (m, n)
        :return: ndarray of shape (m, r)
            Returns the product v @ P^(-1).
        """
        raise NotImplementedError

    def fwd(self, v):
        """
        Evaluates the regularization operator at v.
        :param v: ndarray of shape (n,).
        :return: ndarray of shape (r,).
            Returns P(v)
        """
        raise NotImplementedError

    def inv(self, w):
        """
        Evaluates the inverse regularization operator at w.
        :param w: ndarray of shape (r,).
        :return: ndarray of shape (n,).
            Returns P^(-1)(w)
        """
        raise NotImplementedError


