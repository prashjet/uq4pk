from experiment_kit import *

from experiment7 import Supertest7


# ------------------------------------------------------------------- RUN
name = "experiment7"
logger = Logger(f"experiment7.log")
logger.activate()

list_of_f = get_f("../data20")
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
experiment7 = make_experiment(name=name,
                              supertest = Supertest7(),
                              name_list=snr_list,
                              data_list_list=data_list_list)
experiment7.do_all()

logger.deactivate()