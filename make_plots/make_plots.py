"""
Creates all plots using the information in "out".
Plots are saved in "plots".
"""


from pathlib import Path

from src.blob_detection import plot_blob_detection
from src.fci import plot_fci
from src.intro import plot_intro
from src.m54 import plot_m54
from src.svd_mcmc import plot_svd_mcmc

MODE = "test"
# "test" just for checking that code runs, "base" for faster computations, "final" for final computations.

if MODE == "final":
    OUT = Path("out/out_test")
    PLOTS = Path("plots/plots_test")
elif MODE == "base":
    OUT = Path("out/out_base")
    PLOTS = Path("plots/plots_base")
elif MODE == "final":
    OUT = Path("out/out_final")
    PLOTS = Path("plots/plots_final")
else:
    raise ValueError("Only available modes are 'test', 'base' and 'final'.")


plot_intro(src=OUT, out=PLOTS)
plot_blob_detection(src=OUT, out=PLOTS)
plot_fci(src=OUT, out=PLOTS)
plot_m54(src=OUT, out=PLOTS)
plot_svd_mcmc(src=OUT, out=PLOTS)