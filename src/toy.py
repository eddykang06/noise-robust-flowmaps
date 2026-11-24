"""The 2-D toy distribution: a two-armed spiral, scaled to unit variance per coordinate."""

import math

import torch

from src.data import Batch


def spiral(n: int, gen: torch.Generator) -> torch.Tensor:
    """Two-armed spiral with small radial jitter."""
    t = 3 * math.pi * torch.rand(n, generator=gen).sqrt()
    arm = torch.randint(0, 2, (n,), generator=gen) * 2 - 1
    r = t + 0.1 * torch.randn(n, generator=gen)
    xy = arm[:, None] * torch.stack([r * t.cos(), r * t.sin()], 1)
    return xy / 4.71  # per-coordinate std: sqrt((E[t^2] + 0.01) / 2), E[t^2] = 9 pi^2 / 2


TOYS = {"spiral": spiral}


def sample_clean(name: str, n: int, gen: torch.Generator) -> Batch:
    """Fresh clean samples; used for the training set and as the evaluation reference."""
    return Batch(x=TOYS[name](n, gen), sigma=torch.zeros(n))
