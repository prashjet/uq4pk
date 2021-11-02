"""
Executes experiment 3.
"""


from experiments.experiment_kit import *

from experiment3 import Experiment3


name = "lci_vs_fci"
logger = Logger(f"experiment3.log")
logger.activate()

list_of_f = get_f("../data5")
snr_list = [2000, 100]
# create data list
data_list_list = []
for snr in snr_list:
    data_list = []
    for f in list_of_f:
        data_list.append(simulate(snr, f))
    data_list_list.append(data_list)
super_test = Experiment3(outname=name, name_list=snr_list, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()