
import shapely.affinity as aff
import shapely.geometry as geom

from uq4pk_fit.gaussian_blob.gaussian_blob import GaussianBlob


eps = 1e-3


def compute_overlap(blob1: GaussianBlob, blob2: GaussianBlob) -> float:
    """
    Computes the relative overlap of two blobs using shapely, i.e.

    .. math::
        o_r = \\frac{A_{intersection}}{\\min(A_1, A_2)}.

    The implementation uses shapely (https://pypi.org/project/Shapely/).

    :param blob1:
    :param blob2:
    :return: The relative overlap, a number between 0 and 1.
    """
    # Create shapely.ellipse objects
    ell1 = _create_ellipse(blob1)
    ell2 = _create_ellipse(blob2)

    # Compute areas of the two ellipses.
    a1 = ell1.area
    a2 = ell2.area
    # Compute intersection area.
    a_intersection = ell1.intersection(ell2).area

    # Compute relative overlap.
    relative_overlap = a_intersection / min(a1, a2)

    # Return relative overlap
    if not 0. - eps <= relative_overlap <= 1. + eps:
        raise RuntimeError(f"Relative overlap = {relative_overlap}")
    return relative_overlap


def _create_ellipse(blob: GaussianBlob):
    """
    Creates a shapely-ellipse object from a Gaussian blob.

    :param blob:
    :return: A shapely ellipse.
    """
    circ = geom.Point(blob.position).buffer(1)
    ellipse = aff.scale(circ, 0.5 * blob.height, 0.5 * blob.width)
    rotated_ellipse = aff.rotate(ellipse, angle=blob.angle)
    return rotated_ellipse
