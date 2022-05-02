"""
Variant of CI computer that computes the filtered credible interval using warm-starts.
"""


import numpy as np
from typing import List
from time import time
import ray

from ..linear_model import CredibleRegion, LinearModel
from ..evaluation import AffineEvaluationFunctional, AffineEvaluationMap
from ..optimization import SOCP
from .distributed_solve import solve_local, solve_remote
from .progress_bar import ProgressBar


NUM_CPU = 7


RTOL = 0.01     # relative tolerance for the cost-constraint
RACC = 1e-2      # relative accuracy for optimization solvers.
USE_RAY = True


class StackComputer:
    """
    Manages the computation of credible intervals from evaluation maps.

    Given a linear model and an evaluation map, the CI computer solves for each evaluation object (loss, x, phi)
    the optimization problems
    min_z / max_z w^\top z s.t. AU z = b - A v, ||C z - d||_2^2 <= e.
    and returns the credible interval [phi(z_min), phi(z_max)].
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, aemap_list: List[AffineEvaluationMap],
                 scale: float):
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._aemap_list = aemap_list
        self._num_scales = len(aemap_list)
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map
        self._abstol = RACC * scale
        self._cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map
        # Initialize ray
        self._num_cpus = NUM_CPU
        self._use_ray = USE_RAY
        self._optno = 2 * self._num_scales * self._dim
        if self._use_ray:
            ray.shutdown()
            ray.init(num_cpus=NUM_CPU)
            self._pb = ProgressBar(self._optno)
            self._actor = self._pb.actor

    def compute_all(self):
        """
        Computes all credible intervals.

        :returns: Object of type :py:class:`CredibleInterval`
        """
        # Create first SOCP.
        aefun_0 = self._aemap_list[0].aef_list[0]
        t0 = time()
        socp = self._create_socp(aefun_0)
        t1 = time()
        print(f"Time for initializing SOCP: {t1 - t0}.")
        # Partition problems.
        pixels_per_cpu = np.ceil(self._dim / self._num_cpus).astype(int)
        n_part = [j * pixels_per_cpu for j in range(self._num_cpus)]
        n_part.append(self._dim)    # Last pixel
        # Create aef list for each individual CPU.
        aef_list_for_cpu = []
        for k in range(self._num_cpus):
            index_set = range(n_part[k], n_part[k+1])
            scale_set = range(self._num_scales) # all scales
            aef_list_k = [[self._aemap_list[j].aef_list[i] for j in scale_set] for i in index_set]
            aef_list_for_cpu.append(aef_list_k)
        lower_result_list = []
        upper_result_list = []
        for k in range(self._num_cpus):
            if self._use_ray:
                socp_id = ray.put(socp)
                lower_result = solve_remote.remote(socp=socp_id, aef_list_list=aef_list_for_cpu[k], actor=self._actor,
                                                   mode="min")
                upper_result = solve_remote.remote(socp=socp_id, aef_list_list=aef_list_for_cpu[k], actor=self._actor,
                                                   mode="max")
                lower_result_list.append(lower_result)
                upper_result_list.append(upper_result)
            else:
                n_pixels = len(aef_list_for_cpu[k])
                n_scales = len(aef_list_for_cpu[k][0])
                lower_result = solve_local(socp=socp, aef_list_list=aef_list_for_cpu[k], mode="min")
                upper_result = solve_local(socp=socp, aef_list_list=aef_list_for_cpu[k], mode="max")
                assert lower_result.shape == (n_scales, n_pixels)
                assert lower_result.shape == (n_scales, n_pixels)
                lower_result_list.append(lower_result)
                upper_result_list.append(upper_result)
        t0 = time()
        if self._use_ray:
            self._pb.print_until_done()
            lower_list = ray.get([lower_result for lower_result in lower_result_list])
            upper_list = ray.get([upper_result for upper_result in upper_result_list])
        else:
            lower_list = [lower_result for lower_result in lower_result_list]
            upper_list = [upper_result for upper_result in upper_result_list]
        t1 = time()
        print(f"Done. Total time = {t1 - t0}s.")

        # Assemble solution.
        lower_stack = np.column_stack(lower_list)
        upper_stack = np.column_stack(upper_list)

        assert np.all(lower_stack <= upper_stack + 1e-5)

        return lower_stack, upper_stack

    # PROTECTED

    def _create_socp(self, aefun: AffineEvaluationFunctional) -> SOCP:
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
            d_z = P_z1.T d_tilde.
            e = e_z - ||P_z1.T d_tilde||_2^2
            P_z [T_z; 0] is the QR decomposition of T U.
            A_z = A U
            b_z= b - A v
            lb_z = [depends on affine evaluation functional]
        """
        w = aefun.w
        u = aefun.u
        v = aefun.v

        # Reformulate the cone constraint for z.
        c_z = self._cregion.t @ u
        g = self._cregion.d_tilde - self._cregion.t @ v
        p_z, t_z0 = np.linalg.qr(c_z, mode="complete")
        k = aefun.zdim
        t_z = t_z0[:k, :]
        p_z1 = p_z[:, :k]
        p_z2 = p_z[:, k:]
        d_tilde_z = p_z1.T @ g
        e_tilde_z = self._cregion.e_tilde - np.sum(np.square(p_z2.T @ g))

        # Reformulate the equality constraint for z.
        if self._cregion.a is not None:
            a_new = self._cregion.a @ u
            b_new = self._cregion.b - self._cregion.a @ v
            # If equality constraint does not satisfy constraint qualification, it is removed.
            satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
            if not satisfies_cq:
                a_new = None
                b_new = None
        else:
            a_new = None
            b_new = None

        # Reformulate the lower-bound constraint for z.
        lb_z = aefun.lb_z(self._cregion.lb)

        # Create SOCP instance
        socp = SOCP(w=w, a=a_new, b=b_new, c=t_z, d=d_tilde_z, e=e_tilde_z, lb=lb_z)
        return socp

    def _print_status(self, i: int, j: int, t_avg, t_avg2):
        """
        Displays the current status of the computation.
        """
        # Estimated solution time.
        n_all = self._dim
        n_remaining = n_all - j
        t_remaining = t_avg * n_remaining
        print("\r", end="")
        print(f"Computing credible interval at scale {i+1}/{self._num_scales} for pixel {j+1}/{self._dim}. "
              f"Average time per pixel: {t_avg:.1f}s. "
              f"Average time in solver: {t_avg2:.1f}s."
              f"Estimated time remaining: {t_remaining:.0f} s.", end=" ")