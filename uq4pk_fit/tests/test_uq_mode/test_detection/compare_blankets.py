
from matplotlib import pyplot as plt
import numpy as np
from skimage import morphology

def look_at_blankets():
    # Load blankets
    blankets = []
    for i in range(1, 6):
        blanket_i = np.loadtxt(f"data/blanket{i}.csv", delimiter=",")
        blankets.append(blanket_i)
    # Visualize slice
    x_span = np.arange(53)
    # Plot all blankets
    for i in range(5):
        blanket_i = blankets[i]
        blanket_slice = blanket_i[6].reshape((53, ))
        plt.plot(x_span, blanket_slice, label=f"Blanket {i}")
    plt.legend()
    plt.show()

def difference_of_blankets():
    map_image = np.loadtxt("data/map.csv", delimiter=",")
    blankets = []
    for i in range(1, 6):
        blanket_i = np.loadtxt(f"data/blanket{i}.csv", delimiter=",")
        blankets.append(blanket_i)
    # Compute difference stack
    dob_list = []
    for i in range(4):
        dob_i = blankets[i] - blankets[i+1]
        dob_list.append(dob_i)
    # Determine local maxima
    dob_stack = np.array(dob_list)
    blanket_stack = np.array(blankets)
    maxima = morphology.local_maxima(image=dob_stack, indices=True, allow_borders=True)
    maxima = np.array(maxima).T
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    plt.imshow(map_image, cmap="gnuplot", aspect="auto")
    for maximum in maxima:
        scale = maximum[0]
        x_coord = maximum[2]
        y_coord = maximum[1]
        ax.add_patch(plt.Circle((x_coord, y_coord), np.sqrt(2) * scale, color='r',
                                fill=False))
    plt.show()

difference_of_blankets()
