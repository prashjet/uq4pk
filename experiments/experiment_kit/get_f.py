
import numpy as np


def get_f(location, numbers: list=None):
    """
    Reads the stored test functions from the csv files and returns them as list of numpy arrays.
    :return: list
    """
    f_list = []
    if numbers is not None:
        for i in numbers:
            f = np.loadtxt(f'{location}/ftest_{i}.csv', delimiter=',')
            f_list.append(f)
    else:
        for i in range(100):
            try:
                f = np.loadtxt(f'{location}/ftest_{i}.csv', delimiter=',')
                f_list.append(f)
            except:
                break
    return f_list