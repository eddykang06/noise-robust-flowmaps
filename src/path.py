"""The linear noising path z_t = (1 - t) x + t eps, shared by objectives and samplers."""

import torch
from torch import Tensor


def bcast(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a per-sample `[B]` tensor to broadcast against `x` of shape `[B, ...]`."""
    return t.view(-1, *[1] * (x.ndim - 1))


def interpolate(x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t = bcast(t, x)
    return (1 - t) * x + t * eps


def sq_err(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """Per-sample squared error summed over the (masked) non-batch dimensions."""
    err = (pred - target) ** 2
    if mask is not None:
        err = err * mask
    return err.flatten(1).sum(1)
