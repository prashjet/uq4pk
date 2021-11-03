
"""
Demo that rectangular credible intervals work better
"""

import tests.unconstrained_problem as testproblem

from uq_mode import *
from tests.testutils import *


# get the fitted unconstrained l2 model
prob = testproblem.get_problem()

# compute local credible intervals
alpha = 0.05
all_components = np.array([0, 1])
part = [all_components]
windows = [all_components, all_components]
plcis = clci(alpha=alpha, partition=part, cost=prob.cost, xmap=prob.xmap)
lcis = lci(alpha=alpha, cost=prob.cost, x_map=prob.xmap, partition=windows)

# visualize flaws
x_plci = get_points(plcis, prob.xmap)
x_lci = get_points(lcis, prob.xmap)

# VISUALIZE TRUE CREDIBLE REGION
# VISUALIZE PEREYRA CREDIBLE REGION
P, h, a = credible_region(alpha=alpha, H=prob.H, y=prob.y, delta=prob.delta, xbar=prob.xbar, xmap=prob.xmap)
circ2 = circle_points(r=np.sqrt(a))
cr2_boundary = prob.xmap[:, np.newaxis] + P @ circ2

plot_result(name="rci_works_better", x_true=prob.xtrue, x_map=prob.xmap, boundary2=cr2_boundary,
            x_lci=x_lci, x_plci=x_plci)