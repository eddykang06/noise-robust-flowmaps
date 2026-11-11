"""Corruption models: map a clean `Batch` to an observed one with per-sample `sigma`."""

import torch

from src.data import Batch
from src.path import bcast


def none(batch: Batch, gen: torch.Generator) -> Batch:
    return batch


def gaussian(batch: Batch, gen: torch.Generator, sigma: float, clean_frac: float) -> Batch:
    """Add N(0, sigma^2) noise to all but a `clean_frac` fraction of samples."""
    n = len(batch)
    noisy = torch.rand(n, generator=gen) >= clean_frac
    s = noisy.float() * sigma
    x = batch.x + bcast(s, batch.x) * torch.randn(batch.x.shape, generator=gen)
    return Batch(x=x, sigma=s, mask=batch.mask, cond=batch.cond)


def gaussian_uniform_var(batch: Batch, gen: torch.Generator, var_max: float,
                         clean_frac: float) -> Batch:
    """Keep a `clean_frac` fraction clean; add N(0, sigma^2) noise to the rest, with
    a variance sigma^2 ~ U[0, var_max] drawn per sample."""
    n = len(batch)
    noisy = torch.rand(n, generator=gen) >= clean_frac
    s = noisy.float() * (var_max * torch.rand(n, generator=gen)).sqrt()
    x = batch.x + bcast(s, batch.x) * torch.randn(batch.x.shape, generator=gen)
    return Batch(x=x, sigma=s, mask=batch.mask, cond=batch.cond)


CORRUPTIONS = {"none": none, "gaussian": gaussian, "gaussian_uniform_var": gaussian_uniform_var}
