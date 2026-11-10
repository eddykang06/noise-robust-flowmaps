"""Flow-map networks with signature `net(z, r, t, cond) -> u`, u shaped like z.

Path convention: z_t = (1 - t) x + t eps, t = 0 data, t = 1 noise. `u` is the
average velocity over [r, t], so z_r = z_t - (t - r) u; r = t gives the
instantaneous velocity.

Networks must be smooth in t: the MeanFlow target contains (t - r) du/dt, so a
time embedding with high frequencies (e.g. DiT's 1000 t scaling) makes the
target explode at initialisation and stalls training.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """For flat data `[B, dim]`; takes t and the interval length t - r as raw inputs."""

    def __init__(self, dim: int, hidden: int, depth: int):
        super().__init__()
        layers = [nn.Linear(dim + 2, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, dim))

    def forward(self, z, r, t, cond=None):
        return self.net(torch.cat([z, t[:, None], (t - r)[:, None]], 1))


MODELS = {"mlp": MLP}
