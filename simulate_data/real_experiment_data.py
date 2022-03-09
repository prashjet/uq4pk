
from copy import deepcopy
import numpy as np
from ppxf import ppxf
from matplotlib import pyplot as plt

import uq4pk_src

from uq4pk_fit.inference import *
from .experiment_data import ExperimentData

THETA_NOISE = 0.05
THETA_V = np.array([146, 3, 1., 0., 0., -0.008, -0.003])
THETA_SCALE = np.array([145, 35, 1., 0.01, 0.01, 0.028, 0.23])


class RealExperimentData(ExperimentData):
    """
    Corresponds to M54 data.
    """

    def __init__(self, simulate: bool = True, normalize: bool = True, snr: float = None):
        # get M54 data
        y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps = self._extract_data()
        f_ppxf = self._do_ppxf(y, y_sd, mask, ssps)
        if normalize:
            # normalize data
            y, y_sd, f_gt, f_ppxf = self._normalize_data(y=y, y_sd=y_sd, f_gt=f_gt, f_ppxf=f_ppxf, fwdop=fwdop,
                                                         theta_guess=theta_guess, mask=mask)
        if simulate:
            theta_sim = theta_guess + theta_sd * np.random.randn(theta_guess.size)
            y, y_sd = self._simulate(f_gt, theta_sim, y_sd, fwdop, snr, mask=mask)
        else:
            # We don't know true theta
            theta_sim = theta_guess

        # Set all the instance variables
        self.name = "M54"
        self.hermite_order = 3
        self.y_unmasked = y
        self.y_sd_unmasked = y_sd
        self.y = y[mask]
        self.y_sd = y_sd[mask]
        self.f_ref = f_gt
        self.theta_ref = theta_sim
        self.theta_sd = theta_sd
        self.mask = mask
        self.snr = np.linalg.norm(self.y) / np.linalg.norm(self.y_sd)
        # Set _ssps and _fwdop variables.
        self._ssps = ssps
        self._fwdop = fwdop

    @property
    def ssps(self):
        return self._ssps

    @property
    def forward_operator(self) -> ForwardOperator:
        return self._fwdop

    def correct_forward_operator(self, continuum_distortion: np.ndarray):
        """
        Corrects the observation operator by multiplying it with a given Legendre polynomial.

        :param continuum_distortion:
        """
        self._ssps.Xw *= (self._ssps.Xw.T * continuum_distortion).T
        self._fwdop = ForwardOperator(ssps=self._ssps, dv=self._ssps.dv, do_log_resample=False, mask=self.mask)

    def legendre_best_fit(self) -> np.ndarray:
        """
        Fits a Legendre polynomial-correction using ppxf.
        :return:
        """
        ssps=self._ssps
        y = self.y_unmasked
        y_sd = self.y_sd_unmasked
        mask = self.mask

        templates = ssps.Xw
        velscale = ssps.dv
        start = [0., 30., 0., 0.]
        bounds = [[-500, 500], [3, 300.], [-0.3, 0.3], [-0.3, 0.3]]
        moments = 4  # 6
        templates = templates[:-1, :]
        truncated_mask = mask[:-1]
        galaxy = y[:-1]
        noise = y_sd[:-1]
        # Perform PPXF fit.
        ppxf_fit_mdegree = ppxf.ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            start=start,
            degree=-1,
            mdegree=10,
            moments=moments,
            bounds=bounds,
            regul=0,
            mask=truncated_mask
        )
        # Plot the best fitting polynomial.
        x = np.linspace(-1, 1, len(galaxy))
        y = np.polynomial.legendre.legval(x, np.append(1, ppxf_fit_mdegree.mpolyweights))
        plt.plot(x, y)
        plt.show()

        # Create continuum distortion
        continuum_distortion = ppxf_fit_mdegree.mpoly
        continuum_distortion = np.concatenate([continuum_distortion, [continuum_distortion[-1]]])
        return continuum_distortion


    @staticmethod
    def _extract_data():
        """
        Extracts data from the m54-dataset.
        :returns y: (m,) array_like
            The real measurement data (masked).
        :returns y_sd: (m,) array_like
            The corresponding vector of standard deviations.
        :returns fwdop: ForwardOperator
            The masked forward operator.
        :returns theta_guess: (7,) array
            A good initial guess for the corresponding LOSVD-parameters.
        :returns theta_sd: (7,) array
            The associated prior standard deviations.
        :returns f_gt: (dim,) array
            The ground truth as a flattened vector, normalized such that
        :returns f_ppxf: (dim,) array
            The MAP estimate from the PPFX software.
        """
        # prepare data
        m54_data = uq4pk_src.data.M54()
        m54_data.logarithmically_resample(dv=50.)
        # read data and noise level
        y = m54_data.y
        y_sd = m54_data.noise_level
        # get mask
        mask = m54_data.mask
        # remove jumps at start and end
        npix_buffer_mask = 20
        mask[:npix_buffer_mask] = False
        mask[-npix_buffer_mask:] = False
        # get initial guess for theta_v
        theta_guess = THETA_V
        # get standard deviations
        theta_sd = THETA_NOISE * THETA_SCALE
        # get "ground truth"
        f_gt = np.flipud(m54_data.ground_truth.T).flatten()
        # theta_v is just the default guess
        # set up the SSPS grid
        ssps = uq4pk_src.model_grids.MilesSSP(
            miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
            imf_string='Ebi1.30',
            lmd_min=None,
            lmd_max=None,
        )
        ssps.resample_spectra(m54_data.lmd)
        # normalise the SSP templates to be light-weighted rather than mass-weighted,
        ssps.Xw /= np.sum(ssps.Xw, 0)
        ssps.dv = m54_data.dv
        ssps.speed_of_light = m54_data.speed_of_light
        # construct the forward operator
        fwdop = ForwardOperator(ssps=ssps, dv=ssps.dv, do_log_resample=False, mask=mask)
        return y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps

    @staticmethod
    def _do_ppxf(y, y_sd, mask, ssps):
        templates = ssps.Xw
        velscale = ssps.dv
        start = [0., 30., 0., 0.]
        bounds = [[-500, 500], [3, 300.], [-0.3, 0.3], [-0.3, 0.3]]
        moments = 4  # 6

        templates = templates[:-1, :]
        truncated_mask = mask[:-1]
        galaxy = y[:-1]
        noise = y_sd[:-1]

        ppxf_fit = ppxf.ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            start=start,
            degree=8,
            moments=moments,
            bounds=bounds,
            regul=0,
            mask=truncated_mask
        )
        f = ppxf_fit.weights
        # bring ppxf-estimate in right format
        f_im = np.reshape(f, ssps.par_dims)
        f_im = np.flipud(f_im)
        # have to flip for some reason
        f = f_im.flatten()
        return f

    @staticmethod
    def _simulate(f, theta, y_sd, fwdop, snr, mask):
        # if snr is not None, y_sd gets rescaled to achieve given snr.
        y = fwdop.fwd(f, theta)
        std_noise = np.random.randn(y.size)
        y_noisy = np.zeros(mask.size)
        if snr is None:
            y_noisy[mask] = y + y_sd[mask] * std_noise
        else:
            # if snr is not None, y_sd gets rescaled to achieve given snr.
            y_sd[mask] = y_sd[mask] * np.linalg.norm(y) / (snr * np.linalg.norm(y_sd[mask]))
            y_noisy[mask] = y + y_sd[mask] * std_noise
        # check
        simulated_snr = np.linalg.norm(y_noisy[mask]) / np.linalg.norm(y_sd[mask] * std_noise)
        print(f"Simulated SNR = {simulated_snr}")
        assert y_noisy.size == mask.size == y_sd.size
        return y_noisy, y_sd

    @staticmethod
    def _normalize_data(y, y_sd, f_gt, f_ppxf, theta_guess, fwdop, mask):
        """
        MAYBE only normalize MASKED value.
        Rescale data so that it fits to a age-metallicity distribution that sums to 1.
        :return:
        """
        # scale f_ppxf to 1
        f_ppxf_norm = f_ppxf / np.sum(f_ppxf)
        # rescale f_gt accordingly
        f_gt_norm = f_gt / np.sum(f_gt)
        # obtain a reference measurement
        y_ref = fwdop.fwd(f_ppxf_norm, theta_guess)
        # compute a scaling factor such that ||y_ref|| = ||y||
        s = np.linalg.norm(y_ref) / np.linalg.norm(y[mask])
        # rescale data and noise
        y_norm = y.copy()
        y_norm[mask] *= s
        y_sd_norm = y_sd.copy()
        y_sd_norm[mask] *= s
        assert np.isclose(np.linalg.norm(y_norm[mask]), np.linalg.norm(y_ref))
        assert not np.isnan(y_norm[mask]).any() and not np.isnan(y_sd_norm[mask]).any()
        # return normalized data
        return y_norm, y_sd_norm, f_gt_norm, f_ppxf_norm