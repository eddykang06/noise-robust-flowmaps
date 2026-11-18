"""2-D toys observed in D dimensions, as in the Pixel MeanFlow toy (arXiv 2601.22158, Sec. 5).

The data lie on the plane spanned by the first two columns of a fixed orthogonal
matrix Q. Working in Q's frame, a sample is padded with D - 2 zeros, corrupted, then
rotated into observed coordinates; corrupting in the frame keeps non-Gaussian
per-coordinate noise (e.g. uniform) the same shape within the plane. A network that
is not given t, such as iMF's, must infer it from z_t: in 2-D it cannot, while in D
dimensions the noise off the plane reveals it.

With D = 2 nothing is padded or rotated, so the toys are unchanged.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F

from src.data import Batch

PLANE = 2  # the toys' own dimension


def basis(dim: int) -> torch.Tensor:
    """Q: a fixed Haar-random orthogonal [dim, dim] matrix, the same for every run."""
    gen = torch.Generator().manual_seed(0)
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=torch.float64))
    return (q * r.diagonal().sign()).float()


def observe(clean: Batch, dim: int, corrupt) -> Batch:
    """Clean 2-D samples corrupted by `corrupt(batch)` and observed in `dim` dimensions."""
    if dim == PLANE:
        return corrupt(clean)
    framed = corrupt(replace(clean, x=F.pad(clean.x, (0, dim - PLANE))))
    return replace(framed, x=framed.x @ basis(dim).T)


def split(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Samples observed in `dim` dimensions, in Q's frame: the plane `[n, 2]` and the
    rest `[n, dim - 2]`. With dim = 2 the samples, of any shape, are returned whole."""
    if dim == PLANE:
        return x, None
    x = x @ basis(dim).to(x)
    return x[:, :PLANE], x[:, PLANE:]
