
import numpy as np


MAXITER = 200


def naive_k_enclosing_box(k: int, points: np.ndarray) -> np.ndarray:
    """
    Finds a small (maybe not the smallest) box that contains k of n given points in R^d (k <= n).
    The box is constructed in such a way that it also contains the mean of all points!

    Parameters
    ---
    k
        The desired number of points.
    points : shape (n, d),
        The points. `n` is the number of points and `d` their dimension.

    Returns
    ---
    array_like, shape (2, d)
        The first row corresponds to the lower bound of the box, the second row corresponds to
        the upper bound, such that the box is given as the set {x: box[0] <= x <= box[1]} in R^d.
    """
    # Check that k <= n and that ``points`` has the correct format.
    assert points.ndim == 2
    n = points.shape[0]
    assert k <= n

    # DETERMINE SHAPE OF THE BOX
    alpha = 0.1
    # cut off the alpha/2 smallest and alpha/2 largest values
    points_sorted = np.sort(points, axis=0)
    lower_cutoff = max(np.floor(0.5 * alpha * n - 1).astype(int), 0)
    upper_cutoff = min(np.ceil((1 - 0.5 * alpha) * n - 1).astype(int), n - 1)
    lower_bounds = points_sorted[lower_cutoff]
    upper_bounds = points_sorted[upper_cutoff]
    box = np.row_stack([lower_bounds, upper_bounds])

    # DETERMINE SIZE VIA BISECTION
    size = 1.  # Relative to original size
    size_low = 0.
    points_inside = _points_inside_box(points, box)
    # First, find size such that at least k points are inside box.
    size_high = 1.
    points_inside_high = points_inside
    i = 0
    while points_inside_high.shape[0] < k:
        size_high = 2 * size_high
        scaled_box = _scale_box(box, size_high)
        points_inside_high = _points_inside_box(points, scaled_box)
        i += 1
        if i > MAXITER:
            print("WARNING: Maximum number of iterations reached.")
            break
    # Then, find ideal size through bisection.
    success = False
    for i in range(MAXITER):
        if points_inside.shape[0] >= k:
            # To many points inside, have to make box smaller.
            size_high = size
            size = 0.5 * (size_high + size_low)
        else:
            # Too few points inside, have to make box larger
            size_low = size
            size *= 2.
        # Rescale the box
        scaled_box = _scale_box(box, size)
        # Determine number of points inside scaled box
        points_inside = _points_inside_box(points, scaled_box)
        if points_inside.shape[0] == k:
            success = True
            break
    if not success:
        print("WARNING: Maximum number of iterations reached.")
        # In case we don't find the ideal box size, take the larger one, so that at least k points are inside.
        scaled_box = _scale_box(box, size_high)
        points_inside = _points_inside_box(points, scaled_box)

    # Shrink the box through min-maxing.
    box_lower = np.min(points_inside, axis=0)
    box_upper = np.max(points_inside, axis=0)

    # Perform some sanity checks on ``lower`` and ``upper``, then return.
    d = points.shape[1]
    assert box_lower.size == d == box_upper.size
    mean = np.mean(points, axis=0)
    mean_inside = np.all(mean >= box_lower) and np.all(mean <= box_upper)
    if not mean_inside:
        print("WARNING: Mean point not inside box.")

    # Enforce box_lower <= box_upper
    box_upper = box_upper.clip(min=box_lower)

    box = np.row_stack([box_lower, box_upper])
    # Check that there are at least k points in box
    n_points_inside = _points_inside_box(points, box).shape[0]
    assert n_points_inside >= k, f"There are only {n_points_inside}/{k} points inside the box."

    return box


def _points_inside_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Returns the subset of points inside a given box.

    Parameters
    ---
    points : shape (n, d)
        Rows correspond to points, columns correspond to coordinates.
    box : shape (2, d)
        The box is determined by the condition box[0] <= x <= box[1].

    Returns
    -------
    points_inside : shape (m, d)
        The points inside the box.
    """
    n, d = points.shape
    assert box.shape == (2, d)

    mask = np.all(points >= box[0], axis=1) & np.all(points <= box[1], axis=1)
    points_inside = points[mask, :]
    return points_inside


def _scale_box(box: np.ndarray, relsize: float) -> np.ndarray:
    """
    Rescales a box to the given size, without changing its center.
    """
    assert box.ndim == 2
    assert box.shape[0] == 2

    center = 0.5 * (box[0] + box[1])
    scaled_box = relsize * (box - center[np.newaxis, :]) + center
    return scaled_box