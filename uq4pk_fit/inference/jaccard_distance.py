
import numpy as np

EPS = np.finfo(float).eps


def mean_jaccard_distance(interval1: np.ndarray, interval2: np.ndarray) -> float:
    """
    Computes the mean Jaccard distance of two multidimensional intervals [a, b], [c, d] \subset R^n:
    d_J([a, b], [c, d]) = 1 - vol(intersection([a, b], [c, d])) / vol(union([a, b], [c, d])).

    :param interval1: Numpy array of shape (n, 2), where the first column corresponds to the lower bound a and the
        second column corresponds to the upper bound b.
    :param interval2: Array of the same shape as `interval1`, corresponding to [c, d].
    :return: A number between 0 and 1, where 0 corresponds to identity and 1 corresponds to disjointness.
    """
    # Check that intervals have same dimension
    if interval1.shape != interval2.shape:
        raise Exception("Intervals must have same dimension.")
    if interval1.ndim != 2:
        raise Exception("Expecting two dimensional intervals.")
    if interval1.shape[1] != 2:
        raise Exception("Intervals must be of the form (n,2).")

    # Compute the volume of the intersection:
    #  Determine the intersection [a_i, b_i] = [max(a, c), min(b, d)].
    a_i = np.max(np.column_stack([interval1[:, 0], interval2[:, 0]]), axis=1)
    b_i = np.min(np.column_stack([interval1[:, 1], interval2[:, 1]]), axis=1)
    #  Compute lengths of intersections
    length_intersection = (b_i - a_i).clip(min=0.)

    # Compute the volume of the union.
    #  Determine the union
    a_u = np.min(np.column_stack([interval1[:, 0], interval2[:, 0]]), axis=1)
    b_u = np.max(np.column_stack([interval1[:, 1], interval2[:, 1]]), axis=1)
    #  Compute volumes of unions
    length_union = b_u - a_u

    # Compute vector of Jaccard distances. If length of union is less than machine precision, return 0.
    d_j = 1 - np.divide(length_intersection, length_union, out=np.zeros_like(length_intersection),
                        where=length_union > EPS)
    # Compute mean
    bar_d_j = np.mean(d_j)

    return bar_d_j
