
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

from uq4pk_fit.blob_detection.scale_normalized_laplacian import scale_normalized_laplacian
from uq4pk_fit.blob_detection.detect_blobs import stack_to_blobs
from uq4pk_fit.blob_detection.significant_blobs.detect_significant_blobs import _match_blobs
from uq4pk_fit.blob_detection.detect_blobs import detect_blobs

SIGMA_MIN = 1
SIGMA_MAX = 15
NSCALES = 12

RATIO = 0.5
rthresh = 0.01
overlap = 0.5


def demo_compute_mapped_pairs():
    # Set test parameters
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / (NSCALES - 1)
    sigmas = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    sigma_list = [np.array([RATIO * sigma, sigma]) for sigma in sigmas]
    scales = [0.5 * s ** 2 for s in sigmas]

    # Load blanket stack
    blanket_list = []
    for i in range(NSCALES):
        blanket = np.loadtxt(f"../data/blanket{i}.csv", delimiter=",")
        blanket_list.append(blanket)
    blanket_stack = np.array(blanket_list)

    # Also need MAP features
    map_im = np.loadtxt("../data/map.csv", delimiter=",")
    map_blobs = detect_blobs(image=map_im, sigma_list=sigma_list)

    laplacian_blanket_stack = scale_normalized_laplacian(blanket_stack, sigma_list, mode="reflect")
    sig_blobs = stack_to_blobs(scale_stack=laplacian_blanket_stack, sigma_list=sigma_list, rthresh=rthresh,
                               max_overlap=overlap)

    # Test the detection of significant blobs.
    mapped_pairs = _match_blobs(map_blobs=map_blobs, significant_blobs=sig_blobs, overlap=overlap)
    # Visualize
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(map_im, cmap="gnuplot", aspect="auto")

    for pair in mapped_pairs:
        b, c = pair
        if c is None:
            w = b.width
            h = b.height
            ax.add_patch(patches.Ellipse((b.x2, b.x1), width=w, height=h, color="red",
                                    fill=False))
        else:
            w1 = b.width
            h1 = b.height
            w2 = c.width
            h2 = c.height
            ax.add_patch(patches.Ellipse((b.x2, b.x1), width=w1, height=h1, color="yellow",
                                    fill=False))
            ax.add_patch(patches.Ellipse((c.x2, c.x1), width=w2, height=h2, color="lime",
                                    fill=False))
    plt.savefig("significant_features.png", bbox_inches="tight")
    plt.show()


demo_compute_mapped_pairs()