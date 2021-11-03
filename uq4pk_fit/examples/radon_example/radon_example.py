"""
A demo of uq_mode for the Radon operator.
"""

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

import uq_mode
import radon_setup as radon
from .params import SCALING, SNR


# load the Radon operator
A = radon.load_radon(SCALING)
# simulate a noisy measurement of the Shepp-Logan phantom
y_noisy, y, image, delta = radon.simulate_measurement(snr=SNR, scaling_factor=SCALING, A=A)
# compute MAP estimate
x_map = radon.compute_map(y_noisy, A, delta)
# -- Compute local credible intervals (as described in Cai et al.)
alpha = 0.05
# Setup filter for  function
m, n = image.shape
# Setup cost-dict
def cost_fun(x):
    return radon.negative_log_posterior(x, y_noisy, A, delta)
def cost_grad(x):
    return radon.negative_log_posterior_gradient(x, y_noisy, A, delta)
cost_dict = {"fun": cost_fun, "grad": cost_grad}
# Setup partition. We use superpixels of size 2x2.
a = 2
b = 2
two_by_two_partition = uq_mode.partition.rectangle_partition(m=m, n=n, a=a, b=b)
# Finally, compute the LCIs
print(" ")
lci = uq_mode.lci.lci(alpha=alpha, cost=cost_dict, x_map=x_map, partition=two_by_two_partition)

# NEXT, COMPUTE LOCAL MEAN CREDIBLE INTERVALS FOR COMPARISON
# Make local mean-filter function. We use a localization frame of width 2 in vertical and horizontal direction.
c = 2
d = 2
local_means = uq_mode.fci.ImageLocalMeans(m, n, a, b, c, d)
print(" ")
lmci = uq_mode.fci.fci(alpha=alpha, cost=cost_dict, x_map=x_map, ffunction=local_means)

# NEXT, COMPUTE FILTERED CREDIBLE INTERVALS WITH EXPONENTIAL KERNEL
exponential_filter = uq_mode.fci.ExponentialFilterFunction(m, n)
print(" ")
fci = uq_mode.fci.fci(alpha=alpha, cost=cost_dict, x_map=x_map, ffunction=exponential_filter)

# VISUALIZE
# Plot ground truth and MAP estimate.
radon.plot_with_colorbar(image)
plt.savefig("lci_vs_fci/truth.png", bbox_inches="tight")
map_image = np.reshape(x_map, (m, n))
radon.plot_with_colorbar(map_image)
plt.savefig("lci_vs_fci/map_estimate.png", bbox_inches="tight")
# Plot local credible intervals.
lci_lower_image = np.reshape(lci[:, 0], (m, n))
lci_upper_image = np.reshape(lci[:, 1], (m, n))
vmax = np.max(lci)
radon.plot_with_colorbar(lci_lower_image, vmax)
plt.savefig("lci_vs_fci/lci_lower.png", bbox_inches="tight")
radon.plot_with_colorbar(lci_upper_image, vmax)
plt.savefig("lci_vs_fci/lci_upper.png", bbox_inches="tight")
# Plot local mean credible intervals
lmci_lower_image = np.reshape(lmci[:, 0], (m, n))
lmci_upper_image = np.reshape(lmci[:, 1], (m, n))
radon.plot_with_colorbar(lmci_lower_image)
plt.savefig("lci_vs_fci/lmci_lower.png", bbox_inches="tight")
radon.plot_with_colorbar(lmci_upper_image)
plt.savefig("lci_vs_fci/lmci_upper.png", bbox_inches="tight")
# Plot FCIs with exponential filter
fci_lower_image = np.reshape(fci[:, 0], (m, n))
fci_upper_image = np.reshape(fci[:, 1], (m, n))
radon.plot_with_colorbar(fci_lower_image)
plt.savefig("lci_vs_fci/fci_lower.png", bbox_inches="tight")
radon.plot_with_colorbar(fci_upper_image)
plt.savefig("lci_vs_fci/fci_lower.png", bbox_inches="tight")

plt.show()