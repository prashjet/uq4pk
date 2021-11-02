from experiments.experiment_kit import *

from experiment6 import Experiment6


# ------------------------------------------------------------------- RUN
name = "lci_vs_fci"
logger = Logger(f"experiment5.log")
logger.activate()

list_of_f = get_f("../data5", numbers=[1])
snr_list = [4000]
# create data list
data_list_list = []
for snr in snr_list:
    data_list = []
    for f in list_of_f:
        data_list.append(simulate(snr, f))
    data_list_list.append(data_list)
super_test = Experiment6(outname=name, name_list=snr_list, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()