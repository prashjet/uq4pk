

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from time import time

from uq4pk_fit.uq_mode.k_enclosing_box.k_enclosing_rectangle import k_enclosing_rectangle


def rectangle_demo(n: int, k: int):
    assert k <= n
    # Generate n points randomly.
    random_points = np.random.randn(n, 2)
    # Compute smallest k-enclosing rectangle.
    t0 = time()
    lb, ub = k_enclosing_rectangle(points=random_points, k=k)
    t1 = time()
    print(f"This took {t1 - t0} seconds.")

    # Evaluate numerically.
    points_in_rectangle = [point for point in random_points if np.all(point >= lb) and np.all(point <= ub)]
    n_points_in_rectangle = len(points_in_rectangle)
    print(f"Rectangle contains {n_points_in_rectangle}/{k}.")
    # Visualize result.
    plt.figure(0)
    axis = plt.axes()
    # Draw points.
    axis.scatter(x=random_points[:, 0], y=random_points[:, 1])
    # Draw rectangle.
    axis.add_patch(Rectangle((lb[0], lb[1]), width=ub[0] - lb[0], height=ub[1] - lb[1], linewidth=1, edgecolor="r",
                             facecolor="none"))
    plt.show()

n = 1 * 1000
k = 1 * 900

rectangle_demo(n, k)