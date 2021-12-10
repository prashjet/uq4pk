
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteLaplacian
from uq4pk_fit.uq_mode.blob_detection.minimize_bumps import minimize_bumps
from uq4pk_fit.uq_mode.blob_detection.cheap_blanket import cheap_blanket

N = 12

# One-time utility.
g = DiscreteLaplacian((12, 53), mode="reflect").mat
for i in range(N):
    print(f"Computing minbump {i+1} / {N}.")
    lower = np.loadtxt(f"data/lower{i}.csv", delimiter=",")
    upper = np.loadtxt(f"data/upper{i}.csv", delimiter=",")
    minbump = minimize_bumps(lb=lower, ub=upper)
    plt.figure(figsize=(6, 2.5))
    plt.imshow(minbump, cmap="gnuplot")
    # Save blanket
    np.savetxt(f"data/minbump{i}.csv", minbump, delimiter=",")
plt.show()