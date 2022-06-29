


class CGNGamma:
    """
    Computes the MAP estimate for a linear Bayesian inverse problem with Gaussian prior and linear constraints and gamma
    prior on the regularization factor, i.e. a problem of the form
        ||Q(y - L x)||_2^2 + alpha * ||P(x - m)||_2^2 - log gamma(alpha),

    """