"""A 2-D toy observed in D dimensions, as in the Pixel MeanFlow toy (arXiv 2601.22158, Sec. 5):
the data lie on the plane spanned by the first two columns of a fixed orthogonal matrix Q.

Exact iMF conditions its network on h = t - r only, so it must read t off z_t; in 2-D it
cannot, in D dimensions the noise energy off the plane gives it away. Samples are
corrupted before the rotation, in the frame where the plane is the first two
coordinates, so per-coordinate noise (e.g. uniform) keeps its shape within the plane.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F

from src.data import Batch


def basis(dim: int) -> torch.Tensor:
    """Q: a fixed Haar-random orthogonal [dim, dim] matrix, shared by every run."""
    gen = torch.Generator().manual_seed(0)
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=torch.float64))
    return (q * r.diagonal().sign()).float()


def pad(batch: Batch, dim: int) -> Batch:
    """2-D samples as the first two coordinates of the frame; the rest are zero."""
    return replace(batch, x=F.pad(batch.x, (0, dim - batch.x.shape[1])))


def observe(batch: Batch) -> Batch:
    """Frame coordinates -> observed coordinates."""
    return replace(batch, x=batch.x @ basis(batch.x.shape[1]).T)


def frame(x: torch.Tensor) -> torch.Tensor:
    """Observed coordinates -> frame coordinates: the plane, then the D - 2 others."""
    return x @ basis(x.shape[1]).to(x)
