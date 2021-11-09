from experiment_kit import *

from experiment9 import Supertest9



name = "experiment9"
logger = Logger(f"experiment9.log")
logger.activate()

# get true measurement
# create data list
simulate = [True, False]
data_list_list = []
for sim in simulate:
    expdata = get_real_data(simulate=sim)
    expdata.setup["simulate"] = simulate
    data_list_list.append([expdata])
experiment9 = make_experiment(name=name,
                              supertest=Supertest9(),
                              name_list=simulate,
                              data_list_list=data_list_list)
experiment9.run()
experiment9.evaluate()
experiment9.plot()
experiment9.make_summary()

logger.deactivate()