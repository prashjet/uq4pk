import numpy as np


class Downsampling:
    """
    A downsampling object is a many-to-one map that maps [dim] onto a subset I of [dim]. Computations
    are then only done for the indices in I, and the results are mapped back to get a result for [dim].
    """
    dim: int    # The dimension of the underlying space.
    rdim: int   # The reduced dimension (i.e. the cardinality of the subset I of [dim]).

    def indices(self) -> np.ndarray:
        """
        Returns the index set I as numpy array.

        :return: Of shape (rdim, ).
        """
        raise NotImplementedError

    def reduce(self, i: int) -> int:
        """
        For an index i in [dim], returns the representing index.

        :param i:
        :return: j, an index.
        """
        raise NotImplementedError

    def downsample(self, x: np.ndarray):
        """
        Takes an array of size self.dim and reduces it to size self.rdim by downsampling.

        :param x: Of shape (dim, ).
        :returns u: Of shape (rdim, ). The downsampled vector.
        """
        raise NotImplementedError

    def enlarge(self, u: np.ndarray):
        """
        Takes an array of size self.rdim and enlarges it to size `self.dim` by inverting the `reduce` operation.

        :param u: Of shape (rdim, ).
        :returns x: Of shape (dim, ). It holds that x[i] = u[j], where j = reduce(i).
        """
        raise NotImplementedError


class NoDownsampling(Downsampling):
    """
    Null object for the Downsampling class.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.rdim = dim

    def indices(self) -> np.ndarray:
        return np.arange(self.rdim)

    def reduce(self, i: int) -> int:
        return i

    def downsample(self, x: np.ndarray):
        return x

    def enlarge(self, u: np.ndarray):
        return u