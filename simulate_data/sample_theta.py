import numpy as np


def sample_theta(q, theta_v):
    """
    Generates a noisy guess of theta_v
    :param q: noise percentage
    :param theta_v: the true value of theta_v
    :return: ndarray (dim,), ndarray (dim,dim)
        The initial guess and the corresponding regularization operator.
    """
    n_theta = theta_v.size
    theta_stdev = np.array([30., 100., 1., .05, .05, .05, .1])
    # create scaled noise
    noise = theta_stdev * np.random.randn(n_theta)
    # scale the noise so that ||scaled_noise|| / ||scaled_theta|| = q
    scaling_factor = np.linalg.norm(theta_v / theta_stdev) * q / np.linalg.norm(noise / theta_stdev)
    # add scaled noise noise
    theta_v_guess = theta_v + scaling_factor * noise
    # scale standard deviation
    noise_stdev = theta_stdev * scaling_factor
    return theta_v_guess, noise_stdev