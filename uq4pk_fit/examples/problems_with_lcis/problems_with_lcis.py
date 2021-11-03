"""
This demo file demonstrates some problems with locally credible intervals.
"""

import numpy as np

from uq4pk_fit.uq_mode import *
from uq4pk_fit.tests.test_uq_mode.testutils import *
from uq4pk_fit.tests.test_uq_mode.unconstrained_problem import get_problem


# get the fitted unconstrained l2 model
prob = get_problem()

# compute Cui's locally credible intervals
alpha = 0.05
part = Partition(dim=2, elements=[np.array([0, 1])])
local_credible = lci(alpha=alpha, model=prob.model, x_map=prob.x_map, partition=part)
ffunction = IdentityFilterFunction(dim=2)
pixelwise_credible = fci(alpha=alpha, x_map=prob.x_map, model=prob.model, ffunction=ffunction)

# visualize lci-points
x_lci = get_points(local_credible, prob.x_map)
# visualize pci_points
x_pci = get_points(pixelwise_credible, prob.x_map)

# VISUALIZE PEREYRA CREDIBLE REGION
model = prob.model
xi = local_credible[0, 1] - prob.x_map[0]
x_xi = local_credible[:, 1]
P, h, a = credible_region(alpha=alpha, H=model.h, y=model.y, Q=model.q.mat, xbar=model.m, xmap=prob.x_map, x_xi=x_xi)
circ = circle_points(r=np.sqrt(a))
cr_boundary = prob.x_map[:, np.newaxis] + P @ circ

plot_result(name="problems_with_lcis", x_true=prob.x_true, x_map=prob.x_map, xi=xi ,boundary2=cr_boundary, x_lci=x_lci,
            x_pci=x_pci)

