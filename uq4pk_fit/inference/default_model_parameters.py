
# DEFAULT PARAMETERS FOR MODEL SETUP

# Rule for the regularization parameter, in dependence of the employed snr
def beta1(snr: float) -> float:
    b = 1e3 * snr
    return b