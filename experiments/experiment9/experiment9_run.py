from experiments.experiment_kit import *

from experiment9 import Experiment9



name = "lci_vs_fci"
logger = Logger(f"experiment9.log")
logger.activate()

# get true measurement
# create data list
simulate = [True, False]
data_list_list = []
for sim in simulate:
    data_list = [get_real_data(simulate=sim)]
    data_list_list.append(data_list)
super_test = Experiment9(outname=name, name_list=simulate, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()