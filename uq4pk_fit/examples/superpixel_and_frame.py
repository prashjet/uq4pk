"""
Plots a framed superpixel.
"""

import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

from uq_mode import FramedRectangles


m = 20  # image height
n = 20  # image width
a = 2   # height of superpixel
b = 2   # width of superpixel
c = 2   # horizontal frame width
d = 3   # vertical frame width
pixelno = 24    # number of the superpixel.
# create the FramedRectangles object with the given parameters
rectangle_window_function = FramedRectangles(m, n, a, b, c, d)
window = rectangle_window_function.window(pixelno)
frame = rectangle_window_function.frame(pixelno)
image_flat = np.zeros(m * n)
# mark the superpixel with a 2, the active frame with a 1, and the remaining pixels with 0
image_flat[frame] += 1
image_flat[window] += 1
image = np.reshape(image_flat, (m, n))
# Make a grid to visualize the individual pixels.
ax = plt.gca()
ax.set_yticks(np.arange(-0.5, m, 1))
ax.set_xticks(np.arange(-0.5, n, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color='w', linestyle='-', linewidth=1)
plt.imshow(image)
plt.savefig(f"superpixel_and_frame", bbox_inches="tight")
plt.show()