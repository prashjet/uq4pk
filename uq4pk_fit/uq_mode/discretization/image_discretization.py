
from .discretization import Discretization, AdaptiveDiscretization


class ImageDiscretization(Discretization):
    """
    Special case of discretization where the space R^dim corresponds to an image space R^(m x dim).
    """
    m: int      # Image height.
    n: int      # Image width.


class AdaptiveImageDiscretization(AdaptiveDiscretization):
    """
    Special case of adaptive discretization where the space R^dim corresponds to an image space R^(m x dim).
    """
    m: int  # Image height.
    n: int  # Image width.