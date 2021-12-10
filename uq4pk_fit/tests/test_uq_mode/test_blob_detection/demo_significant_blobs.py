
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

from uq4pk_fit.uq_mode.blob_detection.significant_blobs import _compute_mapped_pairs, RTHRESH
from uq4pk_fit.uq_mode.blob_detection.detect_features import detect_features

SIGMA_MIN = 1
SIGMA_MAX = 15
NSCALES = 12

RATIO = 0.5


def demo_compute_mapped_pairs():
    # Set test parameters
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / (NSCALES - 1)
    sigmas = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    scales = [0.5 * s ** 2 for s in sigmas]

    # Load blanket stack
    blanket_list = []
    for i in range(NSCALES):
        blanket = np.loadtxt(f"data/minbump{i}.csv", delimiter=",")
        blanket_list.append(blanket)
    blanket_stack = np.array(blanket_list)

    # Also need MAP features
    map_im = np.loadtxt("data/map.csv", delimiter=",")
    map_features = detect_features(image=map_im, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, num_sigma=NSCALES - 2,
                                   rthresh=RTHRESH, ratio=RATIO)

    # Test the detection of significant blobs.
    mapped_pairs = _compute_mapped_pairs(blanket_stack=blanket_stack, map_features=map_features,
                                         resolutions=scales, ratio=RATIO)
    # Visualize
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(map_im, cmap="gnuplot", aspect="auto")
    for pair in mapped_pairs:
        b, c = pair
        if c is None:
            s_x, s_y, y, x = b
            w = 2 * np.sqrt(2) * s_x
            h = 2 * np.sqrt(2) * s_y
            ax.add_patch(patches.Ellipse((x, y), width=w, height=h, color="red",
                                    fill=False))
        else:
            s_x1, s_y1, y1, x1 = b
            s_x2, s_y2, y2, x2 = c
            w1 = 2 * np.sqrt(2) * s_x1
            h1 = 2 * np.sqrt(2) * s_y1
            w2 = 2 * np.sqrt(2) * s_x2
            h2 = 2 * np.sqrt(2) * s_y2
            ax.add_patch(patches.Ellipse((x1, y1), width=w1, height=h1, color="yellow",
                                    fill=False))
            ax.add_patch(patches.Ellipse((x2, y2), width=w2, height=h2, color="lime",
                                    fill=False))
    plt.savefig("significant_features.png", bbox_inches="tight")
    plt.show()


demo_compute_mapped_pairs()