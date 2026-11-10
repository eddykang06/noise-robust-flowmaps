"""iMF trained on partly noisy data with an ambient target (ambient diffusion,
Daras et al., applied to the iMF velocity regression).

A sample observed as y = x + sigma * eta (sigma = 0 if clean) sits at path time
t_n = sigma / (1 + sigma). For t > t_n it is noised to the same marginal as clean
data, z_t = (1 - t) y + t sqrt(1 - c) e with c = sigma^2 (1 - t)^2 / t^2. Then
x_hat = (y - c z_t / (1 - t)) / (1 - c) satisfies E[x_hat | z_t] = E[x | z_t], so
v_hat = (z_t - x_hat) / t is an unbiased target for the velocity and replaces e - x in
the iMF loss, whose minimiser it leaves unchanged. Expanding z_t, the target is
v_hat = sqrt(1 - c) (1 + c / ((1 - t)(1 - c))) e - y, computed in that form to avoid
cancellation at small t. Noisy samples with t <= t_n + t_margin are masked out
(the target's variance grows as 1 / (1 - c)).
For sigma = 0 everything reduces to iMF. Unconditional only (no guidance).
"""

import torch

from src.data import Batch
from src.objectives.imf import adp_wt_fn, compound, sample_tr, v_cond_fn
from src.path import bcast, sq_err


def ambient_noising(y, sigma, t, e):
    """z_t from an observation at noise level sigma, and the unbiased velocity target."""
    t_ = bcast(t, y)
    c = (bcast(sigma, y) * (1 - t_) / t_) ** 2
    a = (1 - c).clamp_min(0).sqrt()
    z = (1 - t_) * y + t_ * a * e
    return z, a * (1 + c / ((1 - t_) * (1 - c))) * e - y


def loss(net, batch: Batch, P_mean: float, P_std: float, data_proportion: float,
         norm_p: float, norm_eps: float, t_margin: float):
    y, sigma = batch.x, batch.sigma
    n, device = len(y), y.device
    labels = torch.full((n,), net.num_classes, device=device)
    t, r, _ = sample_tr(n, P_mean, P_std, data_proportion, device)
    valid = (sigma == 0) | (t > sigma / (1 + sigma) + t_margin)
    z, v_hat = ambient_noising(y, sigma, t, torch.randn_like(y))
    v_hat = torch.where(bcast(valid, y), v_hat, 0.0)  # masked targets may be inf; keep grads finite

    omega = torch.ones(n, device=device)
    t_min, t_max = torch.zeros(n, device=device), torch.ones(n, device=device)
    with torch.no_grad():
        v_c = v_cond_fn(net, z, t, omega, labels)
    V, v = compound(net, z, t, r, omega, t_min, t_max, labels, v_c)
    loss_u = adp_wt_fn(sq_err(V, v_hat, batch.mask), norm_p, norm_eps)
    loss_v = adp_wt_fn(sq_err(v, v_hat, batch.mask), norm_p, norm_eps)
    per_sample = torch.where(valid, loss_u + loss_v, 0.0)
    return per_sample.mean(), {"valid": valid.float().mean()}
