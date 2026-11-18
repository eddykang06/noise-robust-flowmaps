"""The linear noising path z_t = (1 - t) x + t e and helpers shared across modules."""

import torch
from torch import Tensor


def bcast(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a per-sample `[B]` tensor to broadcast against `x` of shape `[B, ...]`."""
    return t.view(-1, *[1] * (x.ndim - 1))


def interpolate(x: Tensor, e: Tensor, t: Tensor) -> Tensor:
    t = bcast(t, x)
    return (1 - t) * x + t * e


def expand_mask(mask: Tensor, x: Tensor) -> Tensor:
    """A `Batch.mask`, which covers a leading part of `x`'s shape, viewed to broadcast against `x`."""
    return mask.view(*mask.shape, *[1] * (x.ndim - mask.ndim))


def sq_err(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """Per-sample squared error summed over the non-batch dimensions of the real entries."""
    err = (pred - target) ** 2
    if mask is not None:
        err = err * expand_mask(mask, err)
    return err.flatten(1).sum(1)


def mean_sq_err(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """Squared error averaged over the real entries, for logging."""
    err = (pred - target) ** 2
    if mask is None:
        return err.mean()
    mask = expand_mask(mask, err).expand_as(err)
    return (err * mask).sum() / mask.sum()
