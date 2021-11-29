
from experiment_kit import *

from experiments.autodetect_test.autodetect_test import AutodetectSupertest


# ------------------------------------------------------------------- RUN
name = "autodetect"
logger = Logger(f"autodetect.log")
logger.activate()

numbers = [3]
list_of_f = get_f("data5", numbers=numbers)
snr_list = [2000]
# create data list
data_list_list = []
for snr in snr_list:
    data_list = []
    i = 0
    for f in list_of_f:
        data_list.append(simulate(name=f"f{numbers[i]}", snr=snr, f_im=f))
        i += 1
    data_list_list.append(data_list)
autodetect_experiment = make_experiment(name=name,
                                        supertest = AutodetectSupertest(),
                                        name_list=snr_list,
                                        data_list_list=data_list_list)
autodetect_experiment.plot()

logger.deactivate()