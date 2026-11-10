"""iMF sampling, ported from the official `generate` / `sample_one_step`:
a uniform grid from t = 1 to t = 0, each step z_r = z_t - (t - r) u(z_t, t, t - r)."""

import torch

from src.path import bcast


@torch.no_grad()
def imf(net, eps, steps: int, y, omega: float, t_min: float, t_max: float):
    """`y`: class labels, or None for the null class; `omega`, `[t_min, t_max]`: guidance."""
    n = len(eps)
    full = lambda v: torch.full((n,), v, dtype=eps.dtype, device=eps.device)
    y = torch.full((n,), net.num_classes, device=eps.device) if y is None else y
    ts = torch.linspace(1.0, 0.0, steps + 1, dtype=eps.dtype, device=eps.device)
    z = eps
    for t, r in zip(ts[:-1], ts[1:]):
        t, r = t.expand(n), r.expand(n)
        u = net(z, t, t - r, full(omega), full(t_min), full(t_max), y)[0]
        z = z - bcast(t - r, z) * u
    return z


SAMPLERS = {"imf": imf}
