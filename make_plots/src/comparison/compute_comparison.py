"""
Runs all necessary computations.
"""


from pathlib import Path


from ..mock import load_experiment_data
from .parameters import DATAFILE1, DATAFILE2, DATAFILE3, OUT1, OUT2, OUT3
from .compute_fcis_mcmc import compute_fcis_mcmc
from .make_mcmc_samples import make_mcmc_samples
from .compute_fcis_optimization import compute_fcis_optimization


def compute_comparison(mode: str, out: Path):
    # snr = 1000
    data1 = load_experiment_data(str(DATAFILE1))
    # snr = 100
    data2 = load_experiment_data(str(DATAFILE2))
    # snr = 10
    data3 = load_experiment_data(str(DATAFILE3))
    # Set up stuff.
    setting1 = {"data": data1, "out": out / OUT1}
    setting2 = {"data": data2, "out": out / OUT2}
    setting3 = {"data": data3, "out": out / OUT3}
    setting_list = [setting1, setting2, setting3]
    for setting in setting_list:
        data = setting["data"]
        out = setting["out"]
        print("Computing for " + str(out))
        make_mcmc_samples(mode=mode, data=data, out=out)
        compute_fcis_mcmc(data=data, out=out)
        compute_fcis_optimization(mode=mode, data=data, out=out)