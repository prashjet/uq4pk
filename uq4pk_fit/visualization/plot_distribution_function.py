
from matplotlib import pyplot as plt



def plot_distribution_function(image, ssps = None, show=False, savefile: str = None, vmax=None, vmin=None, ):
    """
    Plots the age-metallicity distribution function with a colorbar on the side that
    shows which color belongs to which value.
    :param image: The age-metallicity distribution as 2-dimensional numpy array.
    :param ticks:
    :param savefile: If provided, stores plot in given location.
    :param vmax:
    :param vmin:
    :param show: If False, the plot is not shown.
    """
    cmap = plt.get_cmap("gnuplot")
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = plt.imshow(image, vmax=vmax, vmin=vmin, cmap=cmap, extent=(0, 1, 0, 1), aspect="auto")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("density")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Metallicity [Z/H]")
    if ssps is not None:
        ticks = [ssps.t_ticks, ssps.z_ticks, ssps.img_t_ticks, ssps.img_z_ticks]
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_yticks(img_z_ticks)
        ax.set_yticklabels(z_ticks)
    # if savename is provided, save .csv and image
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')
    if show: plt.show()
    plt.close()