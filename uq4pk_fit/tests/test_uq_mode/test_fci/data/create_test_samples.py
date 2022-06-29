
import numpy as np

n = 25000
d = 3

samples = np.random.randn(n, d)
np.save(arr=samples, file="samples.npy")