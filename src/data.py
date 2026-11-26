"""Batches and minibatch streams.

A `Batch` carries any data modality: `x` is `[B, ...]` (toys `[B, 2]`, images
`[B, C, H, W]`, proteins `[B, L, 3]`), `mask` marks real entries in padded
data with 1 and padding with 0, over a leading part of `x`'s shape (proteins
`[B, L]`: one entry per residue), `sigma` is each sample's observation-noise
level (0 for clean) and `cond` is optional conditioning. `x_std`, if set, is the per-element std
of a posterior around `x` (ImageNet's VAE posterior): training draws x + x_std * eps afresh each
step (`resample`).
"""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # for data preparation and metrics; training needs CUDA


@dataclass
class Batch:
    x: Tensor
    sigma: Tensor
    mask: Tensor | None = None
    cond: Tensor | None = None
    x_std: Tensor | None = None

    def map(self, fn) -> "Batch":
        """Apply `fn` to every tensor field that is set."""
        return replace(self, **{f.name: fn(v) for f in fields(self)
                                if (v := getattr(self, f.name)) is not None})

    def __getitem__(self, idx) -> "Batch":
        return self.map(lambda v: v[idx])

    def to(self, device) -> "Batch":
        return self.map(lambda v: v.to(device))

    def resample(self) -> "Batch":
        """A fresh draw x + x_std * eps from the posterior; itself (no randomness used) without one."""
        if self.x_std is None:
            return self
        return replace(self, x=self.x + self.x_std * torch.randn_like(self.x), x_std=None)

    def __len__(self) -> int:
        return len(self.x)


def batches(data: Batch, batch_size: int):
    """Infinite stream of random minibatches from an in-memory dataset."""
    while True:
        yield data[torch.randint(len(data), (batch_size,), device=data.x.device)]
