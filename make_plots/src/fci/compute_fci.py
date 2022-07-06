
from pathlib import Path

from .make_mcmc_samples import make_mcmc_samples
from .compute_pci_fci_from_mcmc import compute_pci_fci_from_mcmc


def compute_fci(mode: str, out: Path):
    make_mcmc_samples(mode=mode, out=out)
    compute_pci_fci_from_mcmc(mode=mode, out=out)
