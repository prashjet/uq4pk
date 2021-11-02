from experiments.experiment_kit import *

from experiment8 import Experiment8


name = "out"
logger = Logger(f"experiment8.log")
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
super_test = Experiment8(outname=name, name_list=snr_list, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()

