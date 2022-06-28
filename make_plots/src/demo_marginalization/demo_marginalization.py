"""
This is a demo of the marginalization.
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from uq4pk_fit.inference import StatModel
from uq4pk_fit.inference import LightWeightedForwardOperator
import uq4pk_src
from simulate_data import load_experiment_data
from src.experiment_data.experiment_parameters import LMD_MIN, LMD_MAX, DV


DATA = Path("../experiment_data/snr2000")
REGPARAM = 2000 * 1e3


data = load_experiment_data(str(DATA))
ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
forward_operator = LightWeightedForwardOperator(theta=data.theta_ref, ssps=ssps, dv=DV)
y = data.y
y_sd = data.y_sd
model = StatModel(y=y, y_sd=y_sd, forward_operator=forward_operator)
model.beta1 = REGPARAM
# Fix theta at true value
model.fix_theta_v(indices=np.arange(model.dim_theta), values=data.theta_ref)
true_mass = np.sum(data.f_true)
model.normalize(mass=true_mass)
# MODEL FITTING
# Fit the model by computing MAP.
fitted_model = model.fit()
f_map = fitted_model.f_map.reshape(12, 53)
# Compute marginalized credible intervals
lower, upper = fitted_model.marginal_credible_intervals(alpha=0.05, axis=0)
# Also compute age-marginal for MAP.
marginal_map = np.sum(f_map, axis=0)
# Visualize.
x_span = np.arange(lower.size)
plt.plot(x_span, lower, label="lower")
plt.plot(x_span, upper, label="upper")
plt.plot(x_span, marginal_map, label="MAP")
plt.legend()
plt.show()
