
import numpy as np

from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction

from model_fit.utils import plot_with_colorbar


def yes_or_no(question):
    while "only answer with 'y' or 'n'":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        elif reply[0] == 'n':
            return False

def make_ftest(ntest=3, hermite_order=4):
    """
    WARNING: MAY OVERWRITE IMPORTANT DATA. USE WITH CARE.
    Creates test distribution functions and stores them as csv files 'ftest0.csv', 'ftest1.csv',...
    :param ntest: int
        number of test functions to be stored
    """
    assertion = yes_or_no("WARNING: THIS FUNCTION WILL OVERWRITE IMPORTANT DATA. PROCEED ONLY "
                          "WITH ABSOLUTE CARE. Do you want to proceed? [yes/no]")
    if assertion:
        # generate functions
        op = ObservationOperator(max_order_hermite=4)
        for n in range(ntest):
            f = RandomGMM_DistributionFunction(modgrid=op.ssps).F
            np.savetxt(f"ftest_{n}.csv", f, delimiter=',')
            # also stores the images
            plot_with_colorbar(f, f"ftest_{n}.png")
            print(f"Created 'ftest_{n}.csv'.")
    else:
        print("The function 'make_ftest' was not executed.")