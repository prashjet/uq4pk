"""
Runs all computations in test mode.
Output is written in "out_test".
"""


from pathlib import Path

from src.blob_detection import compute_blob_detection
from src.comparison import compute_comparison
from src.computational_aspects import compute_computational_aspects
from src.fci import compute_fci
from src.m54 import compute_m54


OUT_TEST = Path("out_test")


print("---------- TESTING: BLOB DETECTION ----------")
compute_blob_detection(out=OUT_TEST, mode="test")

print("---------- TESTING: COMPARISON ----------")
compute_comparison(mode="test", out=OUT_TEST)

print("---------- TESTING: COMPUTATIONAL ASPECTS ----------")
compute_computational_aspects(mode="test", out=OUT_TEST)

print("---------- TESTING: FCI----------")
compute_fci(mode="test", out=OUT_TEST)

print("---------- TESTING: M54----------")
compute_m54(mode="test", out=OUT_TEST)
