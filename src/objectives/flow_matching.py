"""Conditional flow matching: regress the instantaneous velocity eps - x (r = t)."""

import torch

from src.data import Batch
from src.path import interpolate, sq_err


def loss(net, batch: Batch):
    x = batch.x
    t = torch.rand(len(x), device=x.device)
    eps = torch.randn_like(x)
    u = net(interpolate(x, eps, t), t, t, batch.cond)
    err = sq_err(u, eps - x, batch.mask)
    return err.mean(), {"sq_err": err.mean()}
