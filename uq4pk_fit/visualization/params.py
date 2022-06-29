
from matplotlib import colors


CMAP = "gnuplot"
COLOR_BLOB = "yellow"
COLOR_SIGNIFICANT = "yellow"
COLOR_OUTSIDE = "magenta"
COLOR_INSIDE = "cyan"

def power_norm(vmax, vmin=0.):
    return colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)