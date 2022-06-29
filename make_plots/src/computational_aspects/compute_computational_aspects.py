

from pathlib import Path

from .compute_localization_plots import compute_localization_plots


def compute_computational_aspects(mode: str, out: Path):
    compute_localization_plots(mode=mode, out=out)
