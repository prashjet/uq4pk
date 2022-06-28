
import numpy as np

from src.m54.plot_m54 import _posterior_predictive_plot


# Simulate data.
n = 3361
t0 = 0.
t1 = 20.
t_diff = t1 - t0
z = np.arange(n)
dt = t_diff / n
x = dt * z
y_fit = np.sin(x) + 2.
noise = 0.1 * np.random.randn(y_fit.size)
y = y_fit + noise
# Create mask
mask = np.full_like(y, True)
mask[500:600] = False
mask[0:100] = False
# Visualize
_posterior_predictive_plot(y=y, y_est=y_fit, mask=mask, file="test.png")