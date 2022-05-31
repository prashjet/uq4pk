
from matplotlib import colors


CMAP = "gnuplot"

def power_norm(vmax, vmin=0.):
    return colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)