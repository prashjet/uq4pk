
import numpy as np

from uq4pk_fit.uq_mode.fci import fci_sampling
from uq4pk_fit.uq_mode.filter import IdentityFilterFunction


def test_fci_sampling():
    alpha = 0.05
    # Load test samples.
    samples = np.load("data/samples.npy")
    n, d = samples.shape    # n is the number of samples, d the dimension of the parameter space
    # Make filter function.
    filter_function = IdentityFilterFunction(dim=d)
    # Compute filtered  credible intervals from samples.
    fci_obj = fci_sampling(alpha=alpha, ffunction=filter_function, samples=samples)
    fci = np.column_stack([fci_obj.lower, fci_obj.upper])

    # First, check that fci has the correct format.
    assert fci.shape == (d, 2)
    # Per definition, (1-\alpha)*100% of the filtered samples must lie inside the fci.
    filtered_samples = [filter_function.evaluate(s) for s in samples]
    filtered_samples = np.array(filtered_samples)
    fci_lb = fci[:, 0]
    fci_ub = fci[:, 1]
    mask = np.all(filtered_samples >= fci_lb, axis=1) & np.all(filtered_samples <= fci_ub, axis=1)
    masked_array = filtered_samples[mask, :]
    n_samples_in_fci = masked_array.shape[0]

    ratio = n_samples_in_fci / n
    print(f"Estimate: {ratio}. Desired: {1 - alpha}")
    assert ratio >= (1 - alpha) # This must hold exactly!!!!