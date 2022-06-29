
from pathlib import Path

from .compute_pixelwise_credible_intervals import compute_pixelwise_credible_intervals
from .compute_filtered_credible_intervals import compute_filtered_credible_intervals


def compute_fci(mode: str, out: Path):
    compute_pixelwise_credible_intervals(mode=mode, out=out)
    compute_filtered_credible_intervals(mode=mode, out=out)
