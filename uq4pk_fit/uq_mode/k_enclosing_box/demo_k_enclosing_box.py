
import numpy as np
from time import time

from uq4pk_fit.uq_mode.k_enclosing_box.k_enclosing_box import k_enclosing_box


def demo_box(d: int, n: int, k: int):
    assert k <= n
    # Generate n points randomly.
    random_points = np.random.randn(n, d)
    # Compute smallest k-enclosing box.
    t0 = time()
    box = k_enclosing_box(points=random_points, k=k)
    lb = box.lb
    ub = box.ub
    t1 = time()
    print(f"This took {t1 - t0} seconds.")

    # Evaluate numerically.
    points_in_rectangle = [point for point in random_points if np.all(point >= lb) and np.all(point <= ub)]
    n_points_in_rectangle = len(points_in_rectangle)
    print(f"Rectangle contains {n_points_in_rectangle}/{k}.")


d = 5
n = 100
k = 90

demo_box(d=d, n=n, k=k)
