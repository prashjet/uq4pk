
from experiment_kit import *

from experiment4 import Supertest4


name = "experiment4"
logger = Logger(f"experiment4.log")
logger.activate()

list_of_f = get_f("../data5", numbers=[1])
snr_list = [2000]
# create data list
data_list_list = []
for snr in snr_list:
    data_list = []
    i = 1
    for f in list_of_f:
        data_list.append(simulate(name=f"f{i}", snr=snr, f_im=f))
        i += 1
    data_list_list.append(data_list)
experiment4 = make_experiment(name=name,
                              supertest=Supertest4(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
logger.deactivate()