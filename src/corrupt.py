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


def per_sample_var(batch: Batch, gen: torch.Generator, var_max: float, clean_frac: float,
                   unit_noise) -> Batch:
    """Keep a `clean_frac` fraction clean; add `sigma * unit_noise` to the rest, with
    a variance sigma^2 ~ U[0, var_max] drawn per sample and zero-mean, unit-variance
    `unit_noise(shape, gen)` per coordinate."""
    n = len(batch)
    noisy = torch.rand(n, generator=gen) >= clean_frac
    s = noisy.float() * (var_max * torch.rand(n, generator=gen)).sqrt()
    x = batch.x + bcast(s, batch.x) * unit_noise(batch.x.shape, gen)
    return Batch(x=x, sigma=s, mask=batch.mask, cond=batch.cond)


def gaussian_uniform_var(batch: Batch, gen: torch.Generator, var_max: float,
                         clean_frac: float) -> Batch:
    """N(0, sigma^2) noise with sigma^2 ~ U[0, var_max] per noisy sample."""
    return per_sample_var(batch, gen, var_max, clean_frac,
                          lambda shape, g: torch.randn(shape, generator=g))


def box_uniform_var(batch: Batch, gen: torch.Generator, var_max: float,
                    clean_frac: float) -> Batch:
    """U[-sqrt(3) sigma, sqrt(3) sigma] noise per coordinate, which has variance sigma^2,
    with sigma^2 ~ U[0, var_max] per noisy sample. `sigma` records that standard deviation,
    as if the noise were Gaussian."""
    return per_sample_var(batch, gen, var_max, clean_frac,
                          lambda shape, g: 3 ** 0.5 * (2 * torch.rand(shape, generator=g) - 1))


CORRUPTIONS = {"none": none, "gaussian": gaussian, "gaussian_uniform_var": gaussian_uniform_var,
               "box_uniform_var": box_uniform_var}
