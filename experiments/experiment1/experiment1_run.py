"""
Executes experiment 1.
"""

from experiment_kit import *

from experiments.experiment1.experiment1 import SuperTest1


name = "test"
logger = Logger(f"experiment1.log")
logger.activate()

list_of_f = get_f("../data5", numbers=[0, 1])
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
experiment1 = make_experiment(name=name,
                              supertest = SuperTest1(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
experiment1.run()
experiment1.evaluate()
experiment1.plot()
experiment1.make_summary()

logger.deactivate()