
from uq4pk_fit.uq_mode.discretization import AdaptiveDiscretization, TrivialDiscretization, CombinedDiscretization, \
    AdaptiveDiscretizationFromList


class HybridDiscretization(AdaptiveDiscretizationFromList):
    """
    The combined discretization for f and theta_v in the case of the nonlinear model.
    """
    def __init__(self, f_discretization: AdaptiveDiscretization, dim_theta: int):
        """
        Given an existing discretization for the f-space, creates a discretization for the combined f-theta space.

        :param f_discretization:
        :param dim_theta: The dimension of theta.
        """
        # First, we turn the f_discretization into a discretization of the f-theta space, by appending trivial
        # discretizations.
        theta_discretization = TrivialDiscretization(dim_theta)
        # Modify the discretizations for f
        discretization_list = []
        for i in range(len(f_discretization.discretizations)):
            discretization_i = f_discretization.discretizations[i]
            new_discretization = CombinedDiscretization(dis1=discretization_i, dis2=theta_discretization)
            discretization_list.append(new_discretization)
        trivial_f_discretization = TrivialDiscretization(dim=f_discretization.dim)
        for i in range(dim_theta):
            theta_discretization_i = CombinedDiscretization(trivial_f_discretization, theta_discretization)
            discretization_list.append(theta_discretization_i)
        # Create new adaptive discretization
        AdaptiveDiscretizationFromList.__init__(self, discretization_list=discretization_list)







