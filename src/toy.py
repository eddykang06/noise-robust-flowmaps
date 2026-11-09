"""2-D toy distributions, scaled to roughly unit variance."""

import math

import torch


def checkerboard(n: int, gen: torch.Generator) -> torch.Tensor:
    """8 alternating unit squares on a 4x4 board centred at the origin."""
    x = 4 * torch.rand(n, generator=gen) - 2
    y = torch.rand(n, generator=gen) + torch.randint(0, 2, (n,), generator=gen) * 2 - 2
    y = y + (x.floor() % 2)
    return torch.stack([x, y], 1) / 1.15


def spiral(n: int, gen: torch.Generator) -> torch.Tensor:
    """Two-armed spiral with small radial jitter."""
    t = 3 * math.pi * torch.rand(n, generator=gen).sqrt()
    arm = torch.randint(0, 2, (n,), generator=gen) * 2 - 1
    r = t + 0.1 * torch.randn(n, generator=gen)
    xy = arm[:, None] * torch.stack([r * t.cos(), r * t.sin()], 1)
    return xy / 4.71  # per-coordinate std: sqrt((E[t^2] + 0.01) / 2), E[t^2] = 9 pi^2 / 2


TOYS = {"checkerboard": checkerboard, "spiral": spiral}
