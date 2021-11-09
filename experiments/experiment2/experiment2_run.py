"""
Executes experiment 2.
"""


from experiment_kit import *

from experiment2 import Supertest2


name = "experiment2"
logger = Logger(f"experiment2.log")
logger.activate()

list_of_f = get_f("../data5", numbers=[0, 1])
snr_list = [2000, 100]
#q_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
q_list = [0.001, 0.01, 0.02]
name_list = []
# create data list
data_list_list = []
for snr in snr_list:
    name_list.append(snr)
    data_list = []
    for q in q_list:
        i = 1
        for f in list_of_f:
            expdata = simulate(name=f"{q}_f{i}", snr=snr, f_im=f, theta_noise=q)
            data_list.append(expdata)
            i += 1
        data_list_list.append(data_list)
experiment2 = make_experiment(name=name,
                             supertest=Supertest2(),
                             name_list=name_list,
                             data_list_list=data_list_list)
experiment2.clean()

logger.deactivate()