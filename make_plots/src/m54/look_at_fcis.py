
from matplotlib import pyplot as plt
import numpy as np
from uq4pk_fit.visualization import plot_distribution_function
from uq4pk_fit.blob_detection.blankets.second_order_blanket import second_order_blanket
from uq4pk_fit.special_operators import DiscreteLaplacian

j = 6

i = 0
lower_stack = np.load("lower_stack.npy")
upper_stack = np.load("upper_stack.npy")
vmin = 0.0
for lower, upper in zip(lower_stack, upper_stack):
    lower_max = lower.max()
    upper_min = upper.min()
    vmax = upper.max()
    blanket = second_order_blanket(lb=lower, ub=upper)
    lower_slice = lower[j]
    upper_slice = upper[j]
    blanket_slice = blanket[j]
    plt.plot(lower_slice)
    plt.plot(upper_slice)
    plt.plot(blanket_slice)
    plt.show()