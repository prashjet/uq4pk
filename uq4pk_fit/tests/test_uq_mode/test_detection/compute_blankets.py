
import numpy as np
from uq4pk_fit.uq_mode.detection.span_blanket import span_blanket


# One-time utility.
for i in range(1, 6):
    lower = np.loadtxt(f"data/lower{i}.csv", delimiter=",")
    upper = np.loadtxt(f"data/upper{i}.csv", delimiter=",")
    blanket = span_blanket(lb=lower, ub=upper)
    # Save blanket
    np.savetxt(f"data/blanket{i}.csv", blanket, delimiter=",")