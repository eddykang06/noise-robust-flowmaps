"""Datasets, selected by `data.name`. Each module provides

    build(cfg, corrupt, gen) -> Batch        the observed training set: clean data, then `corrupt`
    reference(cfg, n, gen) -> Batch          n fresh clean samples (with labels) for evaluation
    figure(cfg, data, ref, generated, path)  a figure of true against generated samples
    ON_DEVICE                                whether the training set is moved to the GPU whole

and image datasets `to_images(x) -> uint8 [B, 3, H, W]` for the image metrics. Prepared files
live under `root()`: `$NRFM_DATA`, by default `data/` in the repository (scripts/prepare_data.py).
A new dataset is one module here plus one entry below.
"""

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("NRFM_DATA", Path(__file__).resolve().parents[2] / "data"))


from src.datasets import imagenet, mnist, spiral  # noqa: E402

DATASETS = {"spiral": spiral, "mnist": mnist, "imagenet": imagenet}
