"""MeanFlow (Geng et al., 2025): learn the average velocity u(z_t, r, t).

Target, from the MeanFlow identity u = v - (t - r) du/dt, where du/dt is the
total derivative along the path, computed as a JVP with tangent (v, 0, 1):

    u_tgt = sg(v - (t - r) (v . d_z u + d_t u)),   v = eps - x.
"""

import torch

from src.data import Batch
from src.path import bcast, interpolate, sq_err


def logit_normal(n: int, mu: float, s: float, device) -> torch.Tensor:
    return torch.sigmoid(mu + s * torch.randn(n, device=device))


def target(net, z, r, t, v, cond=None):
    """The network output u(z, r, t) and its stop-gradient MeanFlow target."""
    u, dudt = torch.func.jvp(lambda z, r, t: net(z, r, t, cond),
                             (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    return u, (v - bcast(t - r, z) * dudt).detach()


def loss(net, batch: Batch, ratio: float, mu: float, s: float, p: float, c: float):
    """`ratio`: fraction of samples with r != t. `mu`, `s`: logit-normal time
    sampling. `p`, `c`: adaptive weight 1 / (err + c)^p."""
    x = batch.x
    n = len(x)
    a, b = logit_normal(n, mu, s, x.device), logit_normal(n, mu, s, x.device)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    r = torch.where(torch.rand(n, device=x.device) < ratio, r, t)

    eps = torch.randn_like(x)
    u, u_tgt = target(net, interpolate(x, eps, t), r, t, eps - x, batch.cond)
    err = sq_err(u, u_tgt, batch.mask)
    weight = 1 / (err.detach() + c) ** p
    return (weight * err).mean(), {"sq_err": err.mean()}
