"""
Computes FCIs via sampling.
"""

import numpy as np
from typing import Sequence, Tuple
from skimage.filters import gaussian

from uq4pk_fit.uq_mode.linear_model import CredibleRegion, LinearModel


def fcis_via_sampling(alpha: float, model: LinearModel, im_map: np.ndarray, sigmas: Sequence[np.array], n_samples: int)\
        -> Tuple[np.ndarray, np.ndarray]:
        m, n = im_map.shape
        x_map = im_map.flatten()
        d = x_map.size
        lb = model.lb
        cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)

        # Create samples w_1, ..., w_N in d dimensions.
        print("Creating samples...")
        w = np.random.randn(d, n_samples)
        # Rescale them so that they have norm^2 = e.
        e = cregion.e_tilde
        w_norms = np.linalg.norm(w, axis=0)
        assert w_norms.size == n_samples
        w = np.sqrt(e) * w / w_norms
        w_squared_norms = np.sum(np.square(w), axis=0)
        assert np.isclose(w_squared_norms, e, rtol=0.1).all()
        # Rescale samples so that they satisfy ||T x - d_tilde||_2^2 = e_tilde.
        t = cregion.t
        print("Inverting T...")
        t_inv = np.linalg.inv(t)
        d_tilde = cregion.d_tilde
        print("Transforming samples...")
        x = np.linalg.solve(t, w + d_tilde[:, np.newaxis])
        # Check that all x satisfy the constraint
        txd_norms = np.sum(np.square(t @ x - d_tilde[:, np.newaxis]), axis=0)
        assert np.isclose(txd_norms, e, rtol=0.1).all()
        # Project samples to nonnegative orthant.
        x = np.clip(a=x, a_min=lb[:, np.newaxis], a_max=None)

        # Compute FCIs on desired scales.
        lower_list = []
        upper_list = []
        i = 1
        n_scales = len(sigmas)
        for sigma in sigmas:
            # Evaluate filter on x
            print(f"Evaluating scale {i}/{n_scales}")
            x_filtered = np.array([gaussian(image=x_i.reshape((m, n)), sigma=sigma, mode="reflect") for x_i in x.T])
            # Determine FCI lower bound.
            fci_low = x_filtered.min(axis=0)
            # Determine upper bound.
            fci_upp = x_filtered.max(axis=0)
            lower_list.append(fci_low)
            upper_list.append(fci_upp)

            # Sanity check: Does filtered MAP lie inside?
            im_map_filtered = gaussian(image=im_map, sigma=sigma, mode="reflect")
            scale = im_map_filtered.max()
            tol = 0.05 * scale
            lower_error = np.max((fci_low - im_map_filtered).clip(min=0.))
            upper_error = np.max((im_map_filtered - fci_upp).clip(min=0.))
            if lower_error > tol:
                print(f"WARNING: Lower error too large: {lower_error/scale}")
            if upper_error > tol:
                print(f"WARNING: Upper error too large: {upper_error/scale}")


            i += 1

        lower_stack = np.array(lower_list)
        upper_stack = np.array(upper_list)
        return lower_stack, upper_stack

