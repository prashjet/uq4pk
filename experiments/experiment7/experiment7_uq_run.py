from experiment_kit import *

from experiment7_uq import Supertest7Uq


name = "experiment7_uq"
logger = Logger(f"experiment7_uq.log")
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
experiment7uq = make_experiment(name=name,
                                supertest = Supertest7Uq(),
                                name_list=snr_list,
                                data_list_list=data_list_list)
experiment7uq.run()
experiment7uq.evaluate()
experiment7uq.plot()
experiment7uq.make_summary()

logger.deactivate()