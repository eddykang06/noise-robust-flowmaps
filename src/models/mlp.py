"""Toy iMF network for flat data `[B, dim]`, built from the official iMF embedders,
RMSNorm, SwiGLU and zero-initialised residual gates; conditioning embeddings are
summed into the input instead of prepended as tokens.

Like imfDiT it sees h = t - r but not t. `condition_on_t` adds a t embedding: a
deviation from iMF for low-dimensional data, where the noise level cannot be read
off z_t as it can off a noisy image."""

import torch
from torch import nn

from src.models.embedder import LabelEmbedder, TimestepEmbedder
from src.models.torch_models import RMSNorm, SwiGLUMlp, TorchLinear


class Block(nn.Module):
    """The iMF transformer block without attention."""

    def __init__(self, hidden: int, weight_init_constant: float):
        super().__init__()
        self.norm = RMSNorm(hidden)
        self.mlp = SwiGLUMlp(hidden, int(hidden * 8 / 3), weight_init_constant=weight_init_constant)
        self.scale = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return x + self.mlp(self.norm(x)) * self.scale


class Head(nn.Module):
    """`depth` blocks and a zero-initialised projection, as in imfDiT's heads."""

    def __init__(self, dim: int, hidden: int, depth: int, weight_init_constant: float):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(hidden, weight_init_constant) for _ in range(depth)])
        self.norm = RMSNorm(hidden)
        self.out = TorchLinear(hidden, dim, weight_init="zeros")

    def forward(self, x):
        return self.out(self.norm(self.blocks(x)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, depth: int, aux_head_depth: int,
                 num_classes: int, weight_init_constant: float = 0.32, condition_on_t: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.t_embedder = TimestepEmbedder(hidden) if condition_on_t else None
        self.x_embedder = TorchLinear(dim, hidden)
        self.h_embedder, self.omega_embedder = TimestepEmbedder(hidden), TimestepEmbedder(hidden)
        self.cfg_t_start_embedder, self.cfg_t_end_embedder = TimestepEmbedder(hidden), TimestepEmbedder(hidden)
        self.y_embedder = LabelEmbedder(num_classes, hidden)
        self.shared = nn.Sequential(*[Block(hidden, weight_init_constant)
                                      for _ in range(depth - aux_head_depth)])
        self.u_head = Head(dim, hidden, aux_head_depth, weight_init_constant)
        self.v_head = Head(dim, hidden, aux_head_depth, weight_init_constant)

    def forward(self, z, t, h, omega, t_min, t_max, y):
        c = (self.h_embedder(h) + self.omega_embedder(1 - 1 / omega) + self.cfg_t_start_embedder(t_min)
             + self.cfg_t_end_embedder(t_max) + self.y_embedder(y))
        if self.t_embedder is not None:
            c = c + self.t_embedder(t)
        x = self.shared(self.x_embedder(z) + c)
        return self.u_head(x), self.v_head(x)

