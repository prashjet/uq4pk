from experiment_kit import *

from experiments.experiment6.experiment6 import Supertest6


# ------------------------------------------------------------------- RUN
def experiment6_run():
    name = "experiment6"
    logger = Logger(f"experiment6.log")
    logger.activate()

    list_of_f = get_f("data5", numbers=[1, 2])
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
    experiment6 = make_experiment(name=name,
                                  supertest = Supertest6(),
                                  name_list=snr_list,
                                  data_list_list=data_list_list)
    experiment6.run()
    experiment6.evaluate()
    experiment6.plot(extra_scale=0.015)
    experiment6.clean()

    logger.deactivate()