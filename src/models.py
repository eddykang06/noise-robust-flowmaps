"""Flow-map networks with signature `net(z, r, t, cond) -> u`, u shaped like z.

Path convention: z_t = (1 - t) x + t eps, t = 0 data, t = 1 noise. `u` is the
average velocity over [r, t], so z_r = z_t - (t - r) u; r = t gives the
instantaneous velocity.
"""

import math

import torch
from torch import nn


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = 1000 * t[:, None] * freqs
    return torch.cat([args.cos(), args.sin()], 1)


class MLP(nn.Module):
    """For flat data `[B, dim]`; conditions on t and the interval length t - r."""

    def __init__(self, dim: int, hidden: int, depth: int, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        layers = [nn.Linear(dim + 2 * t_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, dim))

    def forward(self, z, r, t, cond=None):
        emb = torch.cat([time_embedding(t, self.t_dim), time_embedding(t - r, self.t_dim)], 1)
        return self.net(torch.cat([z, emb], 1))


MODELS = {"mlp": MLP}
