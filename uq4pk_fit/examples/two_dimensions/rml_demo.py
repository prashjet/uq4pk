"""
This demo file demonstrates some problems with locally credible intervals.
"""

import numpy as np
import scipy.stats as scistats

import cgn

import uq_mode.fci
import uq_mode.rml
from tests.testutils import *
from tests.unconstrained_problem import get_problem

# -- COMPUTE CREDIBLE INTERVALS VIA RML
alpha = 0.05
# get the test problem
prob = get_problem()
# make a simple filter function
weights = [np.array([1., 0.]), np.array([0., 1.])]
filter_function = uq_mode.fci.SimpleFilterFunction(dim=2, weights=weights)
# Setup the model
xbar = prob.xbar
regop = cgn.TrivialOperator(dim=xbar.size)
delta = prob.delta
noise_regop = cgn.DiagonalOperator(dim=2, s=1 / delta)
def forward(x):
    return prob.H @ x

def forward_jac(x):
    return prob.H
# Setup model
model = uq_mode.rml.Model(mean_list=[xbar], regop_list=[regop], forward=forward, forward_jac=forward_jac,
              regop_noise=noise_regop)
rmlci = uq_mode.rml.rml_ci(alpha=alpha, ffunction=filter_function, model=model, y=prob.y, nsamples=500)
# visualize flaws
x_ci = get_points(rmlci, prob.xmap)
xmap = prob.xmap
xmap_filter = filter_function.evaluate(xmap)

# VISUALIZE TRUE CREDIBLE REGION
lvl = scistats.chi2.ppf(q=1 - alpha, df=2)
circ = circle_points(r=np.sqrt(lvl))
cr_boundary = prob.xmap[:, np.newaxis] + prob.s_post @ circ

plot_result(name="rml_demo", x_true=prob.xtrue, x_map=prob.xmap, boundary1=cr_boundary,
            x_ci1=x_ci)

