"""
Executes experiment 3.
"""


from experiment_kit import *

from experiment3 import Supertest3


name = "experiment3"
logger = Logger(f"experiment3.log")
logger.activate()

list_of_f = get_f("../data5")
snr_list = [2000, 100]
# create data list
data_list_list = []
for snr in snr_list:
    data_list = []
    i = 1
    for f in list_of_f:
        data_list.append(simulate(name=f"f{i}", snr=snr, f_im=f))
        i += 1
    data_list_list.append(data_list)
experiment3 = make_experiment(name=name,
                              supertest = Supertest3(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
experiment3.do_all()
logger.deactivate()