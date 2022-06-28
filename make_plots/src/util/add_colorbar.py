
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar_to_axis(fig, ax, im):
    """
    Adds colorbar next to given axis.
    :param ax:
    :param im:
    :return:
    """
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Density")


def add_colorbar_to_plot(fig, axes, im):
    """
    Add colorbar to whole plot.
    """
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label("Density")