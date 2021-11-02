


import os
import numpy as np


# load data
def load(name):
    array = np.loadtxt(f"experiment8_out/snr=100_optimization_trial1/{name}.csv", delimiter=",")
    return array


ci_size = load("ci_size")
map = load("filtered_map.png")
truth = load("filtered_truth.png")

diff = np.abs(map - truth)

print(f"max diff = {diff.max()}, min diff = {diff.min()}")

for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]:
    clipped_ratios = ci_size / diff.clip(min=eps)
    cleaned_ratios = (ci_size / diff)[diff > eps]
    t1 = np.mean(clipped_ratios)
    t2 = np.mean(cleaned_ratios)
    t3 = np.median(clipped_ratios)
    print(f"eps = {eps}, clipped tightness = {t1}, cleaned tightness = {t2}, median tightness = {t3}")