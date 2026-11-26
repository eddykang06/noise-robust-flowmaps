"""Corruption models: map a clean `Batch` to an observed one with per-sample `sigma`.

All but a `clean_frac` fraction of samples get zero-mean noise of variance sigma^2 per
coordinate, Gaussian or uniform; `sigma` records its standard deviation, as if the noise
were Gaussian. sigma is either fixed or drawn per sample with sigma^2 ~ U[0, var_max].
"""

from dataclasses import replace

import torch

from src.data import Batch
from src.path import bcast


def normal(shape, gen):
    return torch.randn(shape, generator=gen)


def box(shape, gen):
    """U[-sqrt(3), sqrt(3)] per coordinate: zero mean, unit variance."""
    return 3 ** 0.5 * (2 * torch.rand(shape, generator=gen) - 1)


def add_noise(batch: Batch, gen: torch.Generator, clean_frac: float, draw_sigma, unit_noise) -> Batch:
    """`draw_sigma(n, gen)` gives each sample's noise level if it is noisy."""
    n = len(batch)
    noisy = torch.rand(n, generator=gen) >= clean_frac
    s = noisy.float() * draw_sigma(n, gen)
    x = batch.x + bcast(s, batch.x) * unit_noise(batch.x.shape, gen)
    return replace(batch, x=x, sigma=s)


def fixed(sigma: float):
    return lambda n, gen: sigma


def uniform_var(var_max: float):
    return lambda n, gen: (var_max * torch.rand(n, generator=gen)).sqrt()


def none(batch: Batch, gen: torch.Generator) -> Batch:
    return batch


def gaussian(batch: Batch, gen: torch.Generator, sigma: float, clean_frac: float) -> Batch:
    return add_noise(batch, gen, clean_frac, fixed(sigma), normal)


def uniform(batch: Batch, gen: torch.Generator, sigma: float, clean_frac: float) -> Batch:
    return add_noise(batch, gen, clean_frac, fixed(sigma), box)


def gaussian_uniform_var(batch: Batch, gen: torch.Generator, var_max: float, clean_frac: float) -> Batch:
    return add_noise(batch, gen, clean_frac, uniform_var(var_max), normal)


def box_uniform_var(batch: Batch, gen: torch.Generator, var_max: float, clean_frac: float) -> Batch:
    return add_noise(batch, gen, clean_frac, uniform_var(var_max), box)


CORRUPTIONS = {"none": none, "gaussian": gaussian, "uniform": uniform,
               "gaussian_uniform_var": gaussian_uniform_var, "box_uniform_var": box_uniform_var}
