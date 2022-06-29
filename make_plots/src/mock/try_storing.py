"""
This is a test to check whether storing as a .npy-file takes more storage than storing as a .csv-file.
"""

import numpy as np
import os


n = 123
m = 421
# Create test array
arr = np.random.randn(m, n)
# Store as .npy
np.save(file="test", arr=arr)
# Store as .csv
np.savetxt(fname="test.csv", X=arr)

npy_size = os.path.getsize("test.npy")
csv_size = os.path.getsize("test.csv")

print(f"Numpy size: {npy_size} bytes.")
print(f"CSV size: {csv_size} bytes.")

# ===> .npy is more efficient!