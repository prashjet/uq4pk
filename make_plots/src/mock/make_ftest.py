
import numpy as np

from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction

from uq4pk_fit.visualization import plot_distribution_function


NUMBER_OF_TESTFUNCTIONS = 20


def yes_or_no(question):
    while "only answer with 'y' or 'dim'":
        reply = str(input(question + ' (y/dim): ')).lower().strip()
        if reply[0] == 'y':
            return True
        elif reply[0] == 'dim':
            return False

def make_ftest(ntest):
    """
    WARNING: MAY OVERWRITE IMPORTANT DATA. USE WITH CARE.
    Creates test distribution functions and stores them as csv files 'ftest0.csv', 'ftest1.csv',...
    :param ntest: int
        number of test functions to be stored
    """
    assertion = yes_or_no("WARNING: THIS FUNCTION WILL OVERWRITE IMPORTANT DATA. PROCEED ONLY "
                          "WITH ABSOLUTE CARE. Do you want to proceed?")
    if assertion:
        # generate functions
        op = ObservationOperator(max_order_hermite=4)
        n = 0
        while n < ntest:
            f = RandomGMM_DistributionFunction(modgrid=op.ssps).F
            # show the distribution
            plot_distribution_function(image=f, show=True, ssps=op.ssps)
            # ask the user if he accepts the function
            assertion = yes_or_no("Do you accept the distribution? [y/dim]?")
            if assertion:
                np.savetxt(f"../data/ftest_{n}.csv", f, delimiter=',')
                print(f"Created 'ftest_{n}.csv'.")
                # also store the images
                plot_distribution_function(image=f, savefile=f"../data/ftest_{n}.png", show=False, ssps=op.ssps)
                print(f"Created 'ftest_{n}.png'.")
                n += 1
            else:
                pass
    else:
        print("The function 'make_ftest' was not executed.")