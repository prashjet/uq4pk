"""
Creates all plots using the information in "out".
Plots are saved in "plots".
"""


from pathlib import Path

from src.blob_detection import plot_blob_detection
from src.comparison import plot_comparison
from src.computational_aspects import plot_computational_aspects
from src.fci import plot_fci
from src.m54 import plot_m54


OUT =  Path("out")
PLOTS = Path("plots")


plot_blob_detection(src=OUT, out=PLOTS)
#plot_comparison(src=OUT, out=PLOTS)
#plot_computational_aspects(src=OUT, out=PLOTS)
#plot_fci(src=OUT, out=PLOTS)
#plot_m54(src=OUT, out=PLOTS)