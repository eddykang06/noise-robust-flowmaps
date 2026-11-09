"""Samplers `sample(net, eps, steps, cond) -> x`, integrating from t = 1 to t = 0."""

from functools import partial

import torch

from src.path import bcast


@torch.no_grad()
def integrate(net, eps, steps: int, cond=None, average: bool = True):
    """Step along a uniform grid: z_r = z_t - (t - r) u. With `average`, u is the
    flow map's average velocity u(z_t, r, t); otherwise the instantaneous
    velocity u(z_t, t, t), which is Euler integration."""
    ts = torch.linspace(1, 0, steps + 1, device=eps.device, dtype=eps.dtype)
    z = eps
    for t, r in zip(ts[:-1], ts[1:]):
        t, r = t.expand(len(z)), r.expand(len(z))
        z = z - bcast(t - r, z) * net(z, r if average else t, t, cond)
    return z


SAMPLERS = {"flowmap": partial(integrate, average=True),
            "euler": partial(integrate, average=False)}
