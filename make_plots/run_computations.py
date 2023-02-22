"""
Run from "make_plots" directory with
$ PYTHONPATH=/path/to/uq4pk python3 run_computations.py
"""


from pathlib import Path

from src.blob_detection import compute_blob_detection
from src.fci import compute_fci
from src.svd_mcmc import compute_svd_mcmc
from src.m54 import compute_m54


MODE = "final"
# "test" just for checking that code runs, "base" for faster computations, "final" for final computations.

if MODE == "test":
    OUT = Path("out/out_test")
elif MODE == "base":
    OUT = Path("out/out_base")
elif MODE == "final":
    OUT = Path("out/out_final")
else:
    raise ValueError("Only available modes are 'test', 'base' and 'final'.")


print("---------- COMPUTING: BLOB DETECTION ----------")
compute_blob_detection(mode=MODE, out=OUT)

print("---------- COMPUTING: FCI ----------")
compute_fci(mode=MODE, out=OUT)

print("---------- COMPUTING: M54----------")
compute_m54(mode=MODE, out=OUT)

print("---------- COMPUTING: SVD-MCMC ----------")
compute_svd_mcmc(mode=MODE, out=OUT)
