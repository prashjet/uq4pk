
import numpy as np
import matplotlib.pyplot as plt
import uq4pk_src
from uq4pk_fit.inference import ForwardOperator

m54_data = uq4pk_src.data.M54()
m54_data.logarithmically_resample(dv=50.)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8,8))
ax[0].plot(m54_data.lmd, m54_data.y)
ax[1].plot(m54_data.lmd, m54_data.noise_level)
ax[2].plot(m54_data.lmd, m54_data.mask)

ax[2].set_xlabel('Wavelength [nm]')
ax[0].set_ylabel('Spectrum')
ax[1].set_ylabel('Noise Level')
ax[2].set_ylabel('Mask [0=exclude]')
#plt.show()

def orient_image(img):
    return np.flipud(img.T)

ssps = uq4pk_src.model_grids.MilesSSP(
    miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
    imf_string='Ebi1.30',
    lmd_min=None,
    lmd_max=None,
    )
ssps.resample_spectra(m54_data.lmd)
ssps.dv = m54_data.dv
# normalise the SSP templates to be light-weighted rather than mass-weighted,
ssps.Xw /= np.sum(ssps.Xw, 0)
ssps.speed_of_light = m54_data.speed_of_light

from uq4pk_src.observation_operator import ObservationOperator
G = ObservationOperator(ssps=ssps,
                        dv=ssps.dv,
                        do_log_resample=False)
# example input parametrs
f = np.abs(np.random.uniform(size=(12,53)))
Theta_v = [50, 100, 1, 0, 0, 0.3, 0.1]
theta_ref = [145, 35, 1., 0, 0, 0.28, -0.23]

f_true = orient_image(m54_data.ground_truth)
f_true = f_true / np.linalg.norm(f_true)
print(f"||f_true|| = {np.linalg.norm(f_true)}")
ybar = G.evaluate(f_true, Theta_v)

#plt.plot(ssps.lmd, ybar, '-')
#plt.show()

norm_y_data = np.linalg.norm(m54_data.y[:-1])
norm_y_map = np.linalg.norm(ybar[:-1])
s = norm_y_map / norm_y_data
y_data = m54_data.y * s
print(f"||y_map||/||y_data|| = {norm_y_map / norm_y_data}")


from ppxf import ppxf

templates = ssps.Xw
galaxy = m54_data.y
noise = m54_data.noise_level
velscale = ssps.dv
start = [0., 30., 0., 0.]
#bounds = [[-500,500], [3,300.], [-0.3,0.3], [-0.3,0.3], [-0.3,0.3], [-0.3,0.3]]
bounds = [[-500,500], [3,300.], [-0.3,0.3], [-0.3,0.3]]
moments = 4 # 6
mask = m54_data.mask

npix_buffer_mask = 20
mask[:npix_buffer_mask] = False
mask[-npix_buffer_mask:] = False


theta_guess = [50, 100, 1, 0, 0, 0.3, 0.1]
fwdop = ForwardOperator(ssps=ssps, dv=ssps.dv, do_log_resample=False, mask=mask)
f_ppxf = m54_data.ppxf_map_solution
f_ppxf_normalized = f_ppxf / np.sum(f_ppxf)
# normalize noise in the same way as you normalize it in the experiment
y_ref = fwdop.fwd(f_ppxf_normalized, theta_guess)
# compute the normalization factor for y
normalization_factor = np.linalg.norm(y_ref) / np.linalg.norm(galaxy[mask])
# rescale y so that it has the same scale as y_ref
galaxy =  normalization_factor * galaxy
# also have to rescale y_sd
noise = noise * normalization_factor


templates = templates[:-1,:]
galaxy = galaxy[:-1]
noise = noise[:-1]
mask = mask[:-1]

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
    mask=mask
)
f_fit = ppxf_fit.weights
print(f"f sum = {np.sum(ppxf_fit.weights)}")
# rescale again
f_scaled = f_fit / np.sum(f_fit)
y_ref = fwdop.fwd(f_fit, theta_ref)
normalization_factor = np.linalg.norm(y_ref) / np.linalg.norm(galaxy[mask])
y_scaled = galaxy[mask] * normalization_factor
y_noise = noise[mask] * normalization_factor
rdm = np.linalg.norm((y_scaled - fwdop.fwd(f_scaled, theta_ref)) / y_noise) / np.sqrt(galaxy[mask].size)
print(f"Relative data misfit: {rdm}")
ppxf_fit.plot()
plt.show()
img = np.reshape(ppxf_fit.weights, ssps.par_dims).T
plt.imshow(orient_image(img))
plt.show()


plt.plot(ssps.lmd[:-1][mask], galaxy[mask], '-')
plt.savefig("real_data.png", bbox_inches="tight")
plt.figure()
plt.plot(ssps.lmd[:-1][mask], y_ref, '-')
plt.savefig("prediction.png", bbox_inches="tight")
plt.show()











