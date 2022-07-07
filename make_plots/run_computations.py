"""
Runs all computations..
Output is written in "out".
"""


from pathlib import Path

from src.blob_detection import compute_blob_detection
from src.fci import compute_fci
from src.svd_mcmc import compute_svd_mcmc
from src.m54 import compute_m54


OUT = Path("out")
mode = "final"   # "final" for final computations.


print("---------- COMPUTING: BLOB DETECTION ----------")
#compute_blob_detection(mode=mode, out=OUT)

print("---------- COMPUTING: FCI ----------")
#compute_fci(mode=mode, out=OUT)

print("---------- COMPUTING: SVD-MCMC ----------")
#compute_svd_mcmc(mode=mode, out=OUT)

print("---------- COMPUTING: M54----------")
compute_m54(mode=mode, out=OUT)
