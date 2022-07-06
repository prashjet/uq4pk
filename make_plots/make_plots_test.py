"""
Creates all plots using the information in "out_test".
Plots are saved in "plots_test".
"""


from pathlib import Path

from src.blob_detection import plot_blob_detection
from src.fci import plot_fci
from src.m54 import plot_m54
from src.svd_mcmc import plot_svd_mcmc


OUT_TEST =  Path("out_test")
PLOTS_TEST = Path("plots_test")


#plot_blob_detection(src=OUT_TEST, out=PLOTS_TEST)
#plot_fci(src=OUT_TEST, out=PLOTS_TEST)
plot_svd_mcmc(src=OUT_TEST, out=PLOTS_TEST)
#plot_m54(src=OUT_TEST, out=PLOTS_TEST)