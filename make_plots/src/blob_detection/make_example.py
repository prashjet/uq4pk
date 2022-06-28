
import numpy as np
from uq4pk_fit.uq_mode.filter import GaussianFilterFunction2D
from .parameters import SIGMA_INDEX, SIGMA_LIST


lower_stack = np.load("out/lower_stack_example.npy")
upper_stack = np.load("out/upper_stack_example.npy")
f_map = np.load("out/map.npy")

lower_slice = lower_stack[SIGMA_INDEX]
upper_slice = upper_stack[SIGMA_INDEX]

# Filter MAP
sigma = SIGMA_LIST[SIGMA_INDEX]
gaussian_filter = GaussianFilterFunction2D(m=12, n=53, sigma=sigma, boundary="reflect")
f_map_filtered = gaussian_filter.evaluate(v=f_map.flatten()).reshape(12, 53)

np.save("data/lb.npy", lower_slice)
np.save("data/ub.npy", upper_slice)
np.save("data/filtered_map.npy", f_map_filtered)