"""
Creates all plots using the information in "out_test".
Plots are saved in "plots_test".
"""


from pathlib import Path

from src.blob_detection import plot_blob_detection
from src.comparison import plot_comparison
from src.computational_aspects import plot_computational_aspects
from src.fci import plot_fci
from src.m54 import plot_m54


OUT_TEST =  Path("out_test")
PLOTS_TEST = Path("plots_test")


#plot_blob_detection(src=OUT_TEST, out=PLOTS_TEST)
#plot_comparison(src=OUT_TEST, out=PLOTS_TEST)
#plot_computational_aspects(src=OUT_TEST, out=PLOTS_TEST)
#plot_fci(src=OUT_TEST, out=PLOTS_TEST)
plot_m54(src=OUT_TEST, out=PLOTS_TEST)