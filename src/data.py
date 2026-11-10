"""Batches and datasets.

A `Batch` carries any data modality: `x` is `[B, ...]` (toys `[B, 2]`, images
`[B, C, H, W]`, proteins `[B, L, 3]`), `mask` marks real entries in padded
data, `sigma` is each sample's observation-noise level (0 for clean) and
`cond` is optional conditioning.
"""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from src import toy


@dataclass
class Batch:
    x: Tensor
    sigma: Tensor
    mask: Tensor | None = None
    cond: Tensor | None = None

    def map(self, fn) -> "Batch":
        """Apply `fn` to every tensor field that is set."""
        return replace(self, **{f.name: fn(v) for f in fields(self)
                                if (v := getattr(self, f.name)) is not None})

    def __getitem__(self, idx) -> "Batch":
        return self.map(lambda v: v[idx])

    def to(self, device) -> "Batch":
        return self.map(lambda v: v.to(device))

    def __len__(self) -> int:
        return len(self.x)


def sample_clean(name: str, n: int, gen: torch.Generator) -> Batch:
    """Fresh clean samples; used for the training set and as the evaluation reference."""
    x = toy.TOYS[name](n, gen)
    return Batch(x=x, sigma=torch.zeros(n))


def batches(data: Batch, batch_size: int):
    """Infinite stream of random minibatches from an in-memory dataset."""
    while True:
        yield data[torch.randint(len(data), (batch_size,), device=data.x.device)]
