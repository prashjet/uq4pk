
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np


cbar_aspect = 20
nticks = 5          # Desired number of ticks in colorbar.


def add_colorbar_to_axis(fig, ax, im):
    """
    Adds colorbar next to given axis.
    """
    cbar_width = ax.get_position().height / cbar_aspect
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, cbar_width, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, aspect=cbar_aspect)
    # -- Set ticks.
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label("Density")


def add_colorbar_to_plot(fig, axes, im):
    """
    Add colorbar to whole plot.
    """
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=cbar_aspect)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label("Density")