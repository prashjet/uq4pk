from experiment_kit import *

from experiment8 import Supertest8


name = "experiment8"
logger = Logger(f"experiment8.log")
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
experiment8 = make_experiment(name=name,
                              supertest = Supertest8(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
experiment8.run()
experiment8.evaluate()
experiment8.plot()
experiment8.make_summary()

logger.deactivate()
