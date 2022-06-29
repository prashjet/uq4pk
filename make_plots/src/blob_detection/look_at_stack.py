
import numpy as np
from pathlib import Path

from uq4pk_fit.visualization import plot_distribution_function
from .parameters import LOWER_STACK, UPPER_STACK


src = Path("../../out_test")

lower_stack = np.load(str(src / LOWER_STACK))
upper_stack = np.load(str(src / UPPER_STACK))

for lower, upper in zip(lower_stack, upper_stack):
    vmax = upper.max()
    plot_distribution_function(lower, vmax=vmax, show=True)
    plot_distribution_function(upper, vmax=vmax, show=True)