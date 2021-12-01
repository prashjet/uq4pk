
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature

from uq4pk_fit.uq_mode.detection.span_blanket import span_blanket
from ..plotting.blob_plot import blob_plot


MINSIGMA = 1
MAXSIGMA = 15
RTHRESH = 0.01


def detect(map: np.ndarray, lower: np.ndarray, upper: np.ndarray, scale: float, savedir: str):
    """
    Perform some rudimentary detection.
    """
    # For a first look, perform feature detection on MAP estimate.
    thresh = map.max() * RTHRESH
    map_features = feature.blob_dog(map, min_sigma=MINSIGMA, max_sigma=MAXSIGMA, overlap=.9, threshold=thresh)
    # Make the corresponding plot
    blob_plot(map, map_features, f"{savedir}/map_detect.png")
    # Next, compute the blanket.
    blanket = span_blanket(lb=lower, ub=upper)
    # For checking, make slice plot
    i = 6
    n = lower.shape[1]
    lower_slice = lower[i].reshape((n, ))
    upper_slice = upper[i].reshape((n, ))
    blanket_slice = blanket[i].reshape((n, ))
    xspan = np.arange(n)
    fig = plt.figure()
    plt.plot(xspan, lower_slice)
    plt.plot(xspan, upper_slice)
    plt.plot(xspan, blanket_slice)
    plt.savefig(f"{savedir}/slices.png", bbox_inches="tight")
    # Now, detect features in blanket.
    blanket_features = feature.blob_dog(blanket, min_sigma=scale, max_sigma=MAXSIGMA, overlap=.9,
                                        threshold=thresh)
    # Make blob plot (with resolution visualization).
    blob_plot(image=blanket, blobs=blanket_features, resolution=scale, savename=f"{savedir}/blanket.png")


def remove_treshold(img, features, tresh_rel):
    tresh_abs = tresh_rel * img.max()
    indices_to_delete = []
    for i in range(features.shape[1]):
        coord = features[:, i]
        if img[coord[0], coord[1]] < tresh_abs:
            indices_to_delete.append(i)
    cleaned_features = np.delete(features, indices_to_delete, axis=1)
    return cleaned_features





