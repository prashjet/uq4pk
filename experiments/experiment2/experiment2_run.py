"""
Executes experiment 2.
"""


from experiments.experiment_kit import *

from experiment2 import Experiment2


name = "out"
logger = Logger(f"experiment2.log")
logger.activate()

list_of_f = get_f("../data5")
snr_list = [2000, 100]
q_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
name_list = []
# create data list
data_list_list = []
for snr in snr_list:
    for q in q_list:
        name_list.append(f"{snr}_{q}")
        data_list = []
        for f in list_of_f:
            expdata = simulate(snr, f, theta_noise=q)
            data_list.append(expdata)
        data_list_list.append(data_list)
super_test = Experiment2(outname=name, name_list=name_list, data_list_list=data_list_list)
super_test.perform_tests()

logger.deactivate()