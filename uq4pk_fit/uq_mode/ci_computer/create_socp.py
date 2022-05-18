
import numpy as np

from ..linear_model import CredibleRegion
from ..evaluation import AffineEvaluationFunctional
from ..optimization import SOCP


def create_socp(aefun: AffineEvaluationFunctional, credible_region: CredibleRegion) -> SOCP:
    """
            Creates the SOCP for the computation of the generalized credible interval.
            The constraints
            ||T x - d_tilde||_2^2 <= e_tilde,
            A x = b,
            x >= lb

            are reformulated in terms of z, where x = U z + v:
            ||T_z z - d_z||_2^2 <= e_z,
            A_z z = b_z,
            z >= lb_z,
            where
                g = d_tilde - T v
                d_z = P_z1.T g.
                e = e_z - ||P_z2.T g||_2^2
                P_z [T_z; 0] is the QR decomposition of T U.
                A_z = A U
                b_z = b - A v
                lb_z = [depends on affine evaluation functional]
            """
    w = aefun.w
    u = aefun.u
    v = aefun.v

    # Reformulate the cone constraint for z.
    c_z = credible_region.t @ u
    g = credible_region.d_tilde - credible_region.t @ v
    p_z, t_z0 = np.linalg.qr(c_z, mode="complete")
    k = aefun.zdim
    t_z = t_z0[:k, :]
    p_z1 = p_z[:, :k]
    p_z2 = p_z[:, k:]
    d_tilde_z = p_z1.T @ g
    e_tilde_z = credible_region.e_tilde - np.sum(np.square(p_z2.T @ g))
    # Check that transformation was correct.
    z0 = aefun.z0
    x0 = aefun.x(z0)
    cval_old = credible_region.e_tilde - np.sum(np.square(credible_region.t @ x0 - credible_region.d_tilde))
    cval_new = e_tilde_z - np.sum(np.square(t_z @ z0 - d_tilde_z))

    assert np.isclose(cval_old, cval_new)


    # Reformulate the equality constraint for z.
    if credible_region.a is not None:
        a_new = credible_region.a @ u
        b_new = credible_region.b - credible_region.a @ v
        # If equality constraint does not satisfy constraint qualification, it is removed.
        satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
        if not satisfies_cq:
            a_new = None
            b_new = None
    else:
        a_new = None
        b_new = None

    # Reformulate the lower-bound constraint for z.
    lb_z = aefun.lb_z(credible_region.lb)

    # Create SOCP instance
    x_guess = credible_region.x_map
    socp = SOCP(w=w, a=a_new, b=b_new, c=t_z, d=d_tilde_z, e=e_tilde_z, lb=lb_z, x_guess=x_guess)
    return socp