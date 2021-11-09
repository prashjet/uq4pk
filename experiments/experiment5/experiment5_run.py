"""
Executes experiment 4.
"""


from experiment_kit import *

from experiment5 import Supertest5


name = "experiment5"
logger = Logger(f"experiment5.log")
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
experiment5 = make_experiment(name=name,
                              supertest = Supertest5(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
experiment5.run()
experiment5.evaluate()
experiment5.plot()
experiment5.make_summary()


logger.deactivate()