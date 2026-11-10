"""Improved MeanFlow (iMF) training objective, ported from the official JAX
implementation (github.com/Lyy-iiis/imeanflow, `imf.py`, commit bf60cd7).
Method and function names follow it; defaults are its `configs/default.py`.

Networks have the iMF signature `net(z, t, h, omega, t_min, t_max, y) -> (u, v)`
with h = t - r, u the average velocity and v the auxiliary velocity head, and
expose `num_classes`; label `num_classes` is the null (unconditional) class.
Path: z_t = (1 - t) x + t e, v_t = e - x. The loss regresses the compound
prediction V = u + (t - r) sg(du/dt) and the v-head onto the (guided) velocity,
with du/dt a JVP whose z-tangent is the network's own conditional velocity.

`guidance=False` trains unconditionally: omega = 1, interval [0, 1] and null
labels, which is the official objective at omega = 1 (the guided target then
equals v_t) without its class-conditional machinery.
"""

import torch

from src.data import Batch
from src.path import bcast, interpolate, sq_err


def logit_normal_dist(n, P_mean, P_std, device):
    return torch.sigmoid(torch.randn(n, device=device) * P_std + P_mean)


def sample_tr(n, P_mean, P_std, data_proportion, device):
    """t >= r; the first `data_proportion` of the batch has r = t (flow matching)."""
    t = logit_normal_dist(n, P_mean, P_std, device)
    r = logit_normal_dist(n, P_mean, P_std, device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    fm_mask = torch.arange(n, device=device) < int(n * data_proportion)
    return t, torch.where(fm_mask, t, r), fm_mask


def sample_cfg_scale(n, cfg_beta, device, s_max=7.0):
    """omega in [1, 1 + s_max] with density proportional to omega^-cfg_beta."""
    u = torch.rand(n, device=device)
    s_max = torch.tensor(s_max, device=device)
    if cfg_beta == 1.0:
        return torch.exp(u * torch.log1p(s_max))
    log_base = (1.0 - cfg_beta) * torch.log1p(s_max)
    return torch.exp(torch.log1p(u * torch.expm1(log_base)) / (1.0 - cfg_beta))


def sample_cfg_interval(n, fm_mask, device):
    t_min = 0.5 * torch.rand(n, device=device)
    t_max = 0.5 + 0.5 * torch.rand(n, device=device)
    return torch.where(fm_mask, 0.0, t_min), torch.where(fm_mask, 1.0, t_max)


def v_cond_fn(net, z, t, omega, y):
    """The v-head at h = 0 and interval [0, 1]."""
    zero, one = torch.zeros_like(t), torch.ones_like(t)
    return net(z, t, zero, omega, zero, one, y)[1]


def guidance_fn(net, v_t, z, t, y, fm_mask, omega, t_min, t_max):
    """Guided target v_g and conditional velocity v_c (the JVP tangent)."""
    n, null = len(z), torch.full_like(y, net.num_classes)
    out = v_cond_fn(net, torch.cat([z, z]), torch.cat([t, t]),
                    torch.cat([omega, torch.ones_like(omega)]), torch.cat([y, null]))
    v_c, v_u = out[:n], out[n:]
    v_g_fm = v_t + bcast(1 - 1 / omega, z) * (v_c - v_u)

    w = torch.where((t >= t_min) & (t <= t_max), omega, 1.0)
    v_c = v_cond_fn(net, z, t, w, y)
    v_g = v_t + bcast(1 - 1 / w, z) * (v_c - v_u)
    return torch.where(bcast(fm_mask, z), v_g_fm, v_g), v_c


def cond_drop(net, v_t, v_g, y, class_dropout_prob):
    """Replace a prefix of the batch (of Binomial(n, p) size) by the null class."""
    n = len(y)
    num_drop = (torch.rand(n, device=y.device) < class_dropout_prob).sum()
    drop = torch.arange(n, device=y.device) < num_drop
    return torch.where(drop, net.num_classes, y), torch.where(bcast(drop, v_t), v_t, v_g)


def compound(net, z, t, r, omega, t_min, t_max, y, v_c):
    """V = u + (t - r) sg(du/dt), du/dt along (z, t, r) with tangent (v_c, 1, 0);
    also returns the v-head output of the same forward pass."""
    u, du_dt, v = torch.func.jvp(
        lambda z, t, r: net(z, t, t - r, omega, t_min, t_max, y),
        (z, t, r), (v_c, torch.ones_like(t), torch.zeros_like(r)), has_aux=True)
    return u + bcast(t - r, z) * du_dt.detach(), v


def adp_wt_fn(loss, norm_p, norm_eps):
    return loss / ((loss + norm_eps) ** norm_p).detach()


def loss(net, batch: Batch, P_mean: float, P_std: float, data_proportion: float,
         cfg_beta: float, class_dropout_prob: float, norm_p: float, norm_eps: float,
         guidance: bool):
    x = batch.x
    n, device = len(x), x.device
    y = batch.cond if guidance else torch.full((n,), net.num_classes, device=device)

    t, r, fm_mask = sample_tr(n, P_mean, P_std, data_proportion, device)
    e = torch.randn_like(x)
    z = interpolate(x, e, t)
    v_t = e - x

    # Targets and the JVP tangent carry no gradient: v_g and du/dt are stop-gradient.
    with torch.no_grad():
        if guidance:
            t_min, t_max = sample_cfg_interval(n, fm_mask, device)
            omega = sample_cfg_scale(n, cfg_beta, device)
            v_g, v_c = guidance_fn(net, v_t, z, t, y, fm_mask, omega, t_min, t_max)
            y, v_g = cond_drop(net, v_t, v_g, y, class_dropout_prob)
        else:
            t_min, t_max = torch.zeros(n, device=device), torch.ones(n, device=device)
            omega = torch.ones(n, device=device)
            v_c, v_g = v_cond_fn(net, z, t, omega, y), v_t

    V, v = compound(net, z, t, r, omega, t_min, t_max, y, v_c)
    loss_u = adp_wt_fn(sq_err(V, v_g, batch.mask), norm_p, norm_eps)
    loss_v = adp_wt_fn(sq_err(v, v_g, batch.mask), norm_p, norm_eps)
    logs = {"loss_u": ((V - v_g) ** 2).mean().detach(), "loss_v": ((v - v_g) ** 2).mean().detach()}
    return (loss_u + loss_v).mean(), logs
