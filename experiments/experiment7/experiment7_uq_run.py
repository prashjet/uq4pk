from experiments.experiment_kit import *

from experiment7_uq import Experiment7UQ


name = "out_uq"
logger = Logger(f"experiment7_uq.log")
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
super_test = Experiment7UQ(outname=name, name_list=snr_list, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()