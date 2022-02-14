
import numpy as np

from uq4pk_fit.blob_detection.significant_blobs import fast_second_order_blanket


N = 12


# One-time utility.
for i in range(N):
    print(f"Computing Laplace blanket {i+1} / {N}.")
    lower = np.loadtxt(f"data/lower{i}.csv", delimiter=",")
    upper = np.loadtxt(f"data/upper{i}.csv", delimiter=",")
    blanket = fast_second_order_blanket(lb=lower, ub=upper)
    np.savetxt(f"data/blanket{i}.csv", blanket, delimiter=",")