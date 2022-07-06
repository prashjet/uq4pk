"""
Runs all computations..
Output is written in "out".
"""


from pathlib import Path

from src.blob_detection import compute_blob_detection
from src.comparison import compute_comparison
from src.computational_aspects import compute_computational_aspects
from src.fci import compute_fci
from src.m54 import compute_m54


OUT = Path("out")
mode = "final"   # "final" for final computations.


print("---------- COMPUTING: BLOB DETECTION ----------")
#compute_blob_detection(mode=mode, out=OUT)

print("---------- COMPUTING: COMPARISON ----------")
#compute_comparison(mode=mode, out=OUT)

print("---------- COMPUTING: COMPUTATIONAL ASPECTS ----------")
#compute_computational_aspects(mode=mode, out=OUT)

print("---------- COMPUTING: FCI----------")
#compute_fci(mode=mode, out=OUT)

print("---------- COMPUTING: M54----------")
compute_m54(mode=mode, out=OUT)
