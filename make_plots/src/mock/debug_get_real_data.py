

import numpy as np
from matplotlib import pyplot as plt


from .get_real_data import get_real_data, _extract_data, _normalize_data, _do_ppxf


def debug_get_real_data():
    ed = get_real_data(simulate=False)
    assert np.all(ed.theta_true == ed.theta_guess)
    assert np.isclose(sum(ed.f_ref), 1.)
    assert np.all(ed.forward_operator.dim_y == ed.y.size)
    assert np.all(ed.y.size == ed.y_sd.size)
    plt.imshow(np.reshape(ed.f_ref, (12, 53)))
    plt.figure()
    plt.imshow(np.reshape(ed.f_true, (12, 53)))
    plt.figure()
    plt.show()
    snr = np.linalg.norm(ed.y) / np.linalg.norm(ed.y_sd)
    print(f"snr = {snr}")
    y_ref = ed.forward_operator.fwd(ed.f_ref, ed.theta_guess)
    rdm = np.linalg.norm((ed.y - y_ref) / ed.y_sd) / np.sqrt(ed.y.size)
    print(f"rdm = {rdm}")
    print(f"||y|| / ||y_ref|| = {np.linalg.norm(ed.y) / np.linalg.norm(y_ref)}")
    print(f"||y - y_ref|| / ||y|| = {np.linalg.norm(ed.y) / np.linalg.norm(ed.y - y_ref)}")


def debug_ppxf():
    y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps = _extract_data()
    # compute the baseline solution using ppxf
    f_ppxf = _do_ppxf(y=y, y_sd=y_sd, mask=mask, ssps=ssps)
    norm_y = np.linalg.norm(y[mask])
    y_ref = fwdop.fwd(f_ppxf, theta_guess)
    y_ref2 = fwdop.fwd(f_ppxf / np.sum(f_ppxf), theta_guess)
    y2 = y[mask] / np.sum(f_ppxf)
    norm_ref = np.linalg.norm(y_ref)
    print(f"||y|| / ||y_ref|| = {norm_y / norm_ref}")
    print(f"||y|| / ||y_ref|| = {np.linalg.norm(y2) / np.linalg.norm(y_ref2)}")
    print(f"||y - y_ref|| / ||y|| = {np.linalg.norm(y2 - y_ref2) / np.linalg.norm(y2)}")
    print(f"max(y) = {y2.max()}, max(yref) = {y_ref2.max()}")
    print(f"min(y) = {y2.min()}, min(yref) = {y_ref2.min()}")


def debug_normalize_data():
    y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps = _extract_data()
    f_ppxf = _do_ppxf(y=y, y_sd=y_sd, mask=mask, ssps=ssps)
    y_ref = fwdop.fwd(f_ppxf, theta_guess)
    print(f"||y|| / ||yref|| = {np.linalg.norm(y[mask]) / np.linalg.norm(y_ref)}")
    y_normalized, y_sd_normalized, f_gt_norm, f_ppxf_norm = _normalize_data(y[mask], y_sd[mask], f_gt, f_ppxf)
    y_ref2 = fwdop.fwd(f_ppxf_norm, theta_guess)
    print(f"||y|| / ||yref|| = {np.linalg.norm(y_normalized) / np.linalg.norm(y_ref2)}")
    r1 = np.linalg.norm(y_normalized) / np.linalg.norm(y[mask])
    r2 = np.linalg.norm(f_ppxf_norm) / np.linalg.norm(f_ppxf)
    r3 = np.linalg.norm(y_sd_normalized) / np.linalg.norm(y_sd[mask])
    assert np.isclose(r1, r2)
    assert np.isclose(r2, r3)
    assert np.isclose(np.sum(f_ppxf_norm), 1.)
    print(f"sum(f_gt) = {np.sum(f_gt_norm)}")


def debug_linearity():
    y, y_sd, fwdop, theta_guess, theta_sd, f_gt, f_ppxf = _extract_data()
    # check that scaling f scales y in the same manner
    y_1 = fwdop.fwd(f_ppxf, theta_guess)
    a = 25.
    y_a = fwdop.fwd(a * f_ppxf, theta_guess)
    ratio = np.linalg.norm(y_a) / np.linalg.norm(y_1)
    assert np.isclose(ratio, a)

debug_get_real_data()
