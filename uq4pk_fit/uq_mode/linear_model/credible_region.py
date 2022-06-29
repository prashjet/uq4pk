
from math import exp, sqrt, log
import numpy as np
from scipy.special import lambertw
from typing import Union

from .linear_model import LinearModel


NEW_FORMULA = True  # Set true if you want to use new formula.


class CredibleRegion:
    """
    Makes the Pereyra credible region.
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray):
        self._model = model
        self._cost_map = model.cost(x_map)
        self._cost = model.cost
        self._x_map = x_map
        self._alpha = alpha
        self._dim = model.dim
        tau = sqrt(16 * log(3 / self._alpha) / self._dim)
        if NEW_FORMULA:
            z = - exp(log(alpha) / self._dim - 1)
            if z < - 1 / exp(1):
                print("WARNING: alpha is too large.")
            proportionality_constant = - float(lambertw(z))
        else:
            proportionality_constant = 1. + tau
        self._k_alpha = self._dim * proportionality_constant
        self._gamma_alpha = self._cost_map + self._k_alpha
        self._t, self._d_tilde, self._e_tilde, self._a, self._b, self._lb = self._preprocessing()

    def cost_constraint(self, x):
        return self._gamma_alpha - self._cost(x)

    @property
    def x_map(self) -> np.ndarray:
        """
        The MAP estimate, which always lies inside the credible region.
        """
        return self._x_map

    @property
    def dim(self) -> int:
        """
        The dimension of the parameter space in which the credible region is embedded.
        """
        return self._dim

    @property
    def t(self) -> np.ndarray:
        """
        Returns the reduced matrix T such that the cone constraint is of the form
            ||T f - d_tilde||_2^2 <= e_tilde.

        :return: T
        """
        return self._t

    @property
    def e_tilde(self) -> np.ndarray:
        """
        Returns the transformed vector e_tilde such that the cone constraint is of the form
            ||T f - d_tilde||_2^2 <= e_tilde.
        :return:
        """
        return self._e_tilde

    @property
    def d_tilde(self) -> np.ndarray:
        """
        Returns the transformed vector d_tilde such that the cone constraint is of the form
            ||T f - d_tilde||_2^2 <= e_tilde.
        :return:
        """
        return self._d_tilde

    @property
    def a(self) -> Union[np.ndarray, None]:
        """
        The matrix that defines the equality constraint A x = b.

        :return: Of shape (c, dim), or None if there is no equality constraint.
        """
        return self._a

    @property
    def b(self) -> Union[np.ndarray, None]:
        """
        The right-hand side of the equality constraint A x = b.

        :return: Vector of shape (c, ), or None if there is no equality constraint.
        """
        return self._b


    @property
    def lb(self) -> Union[np.ndarray, None]:
        """
        The bound for the lower-bound constraint f >= lb.

        :return: Of shape (dim, ), or None if there is no lower bound constraint.
        """
        return self._lb

    @property
    def equality_constrained(self) -> bool:
        """
        True, if the credible region is equality-constrained.
        """
        if self._a is not None:
            return True
        else:
            return False

    @property
    def bound_constrained(self) -> bool:
        """
        True if the credible region is bound-constrained.
        """
        if self._lb is not None:
            return True
        else:
            return False

    # PROTECTED

    def _preprocessing(self):
        """
        Computes all entities necessary for the formulation of the constraints
        ||C f - d||_2^2 <= e,
        A f = b
        f >= lb.
        We use a QR decomposition to reduce the cone constraint to
        ||T f - d_tilde||_2^2 <= e_tilde,
        where R is invertible upper triangular.

        :return: t, d_tilde, e_tilde, a, b, lb.
            - t: An invertible upper triangular matrix of shape (dim, dim).
            - d_tilde: An dim-vector.
            - e_tilde: A nonnegative float.
            - a: A matrix of shape (c, dim).
            - b: A c-vector.
            - lb: An dim-vector, representing the lower bound.
        """
        a = self._model.a
        b = self._model.b
        h = self._model.h
        y = self._model.y
        q = self._model.q
        r = self._model.r
        m = self._model.m
        lb = self._model.lb
        n = self._model.dim

        # Assemble the matrix C, the vector d and the RHS e.
        c1 = q.fwd(h)
        c2 = r.mat
        c = np.concatenate([c1, c2], axis=0)
        d1 = q.fwd(y)
        d2 = r.fwd(m)
        d = np.concatenate([d1, d2], axis=0)
        e = 2 * self._gamma_alpha

        # Compute the QR decomposition of C.
        p, t0 = np.linalg.qr(c, mode="complete")
        t = t0[:n, :]
        p1 = p[:, :n]
        p2 = p[:, n:]

        # Compute d_tilde and e_tilde.
        d_tilde = p1.T @ d
        e_tilde = e - np.sum(np.square(p2.T @ d))

        # Return everything in the right-order.
        return t, d_tilde, e_tilde, a, b, lb

