"""
Run from "make_plots" directory with
$ export XLA_FLAGS="--xla_force_host_platform_device_count=[NUM_CPU]"
$ PYTHONPATH=/path/to/uq4pk python3 run_computations.py
"""


from pathlib import Path

from src.blob_detection import compute_blob_detection
from src.fci import compute_fci
from src.svd_mcmc import compute_svd_mcmc
from src.m54 import compute_m54

import jax
print(f"Available devices: {jax.local_device_count()}")


OUT = Path("out")
mode = "base"   # "final" for final computations.


print("---------- COMPUTING: BLOB DETECTION ----------")
#compute_blob_detection(mode=mode, out=OUT)

print("---------- COMPUTING: FCI ----------")
#compute_fci(mode=mode, out=OUT)

print("---------- COMPUTING: SVD-MCMC ----------")
#compute_svd_mcmc(mode=mode, out=OUT)

print("---------- COMPUTING: M54----------")
compute_m54(mode=mode, out=OUT)
