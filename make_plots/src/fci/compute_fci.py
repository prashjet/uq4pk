
from pathlib import Path

from .compute_pixelwise_credible_intervals import compute_pixelwise_credible_intervals
from .compute_filtered_credible_intervals import compute_filtered_credible_intervals
from .make_mcmc_samples import make_mcmc_samples
from .compute_pci_fci_from_mcmc import compute_pci_fci_from_mcmc


with_mcmc = True


def compute_fci(mode: str, out: Path):
    if with_mcmc:
        make_mcmc_samples(mode=mode, out=out)
        compute_pci_fci_from_mcmc(mode=mode, out=out)
    else:
        compute_pixelwise_credible_intervals(mode=mode, out=out)
        compute_filtered_credible_intervals(mode=mode, out=out)
