"""Checks that tests/test_imf_reference.py catches plausible porting mistakes: each mutant
patches one piece of src.objectives.imf, and the comparison with the official JAX
objective must then fail.  python research/experiments/ambient-flowmap/imf_mutants.py"""

import sys

import pytest
import torch

from src.objectives import imf
from src.path import bcast

real = {name: getattr(imf, name) for name in ["compound", "sample_tr", "adp_wt_fn", "sq_err"]}


def compound_no_stop_grad(net, z, t, r, omega, t_min, t_max, y, v_c):
    u, du_dt, v = torch.func.jvp(lambda z, t, r: net(z, t, t - r, omega, t_min, t_max, y),
                                 (z, t, r), (v_c, torch.ones_like(t), torch.zeros_like(r)), has_aux=True)
    return u + bcast(t - r, z) * du_dt, v


def compound_r_tangent(net, z, t, r, omega, t_min, t_max, y, v_c):
    u, du_dt, v = torch.func.jvp(lambda z, t, r: net(z, t, t - r, omega, t_min, t_max, y),
                                 (z, t, r), (v_c, torch.ones_like(t), torch.ones_like(r)), has_aux=True)
    return u + bcast(t - r, z) * du_dt.detach(), v


def sample_tr_suffix(n, P_mean, P_std, data_proportion, device):
    """r = t on the last `data_proportion` of the batch instead of the first."""
    t = imf.logit_normal_dist(n, P_mean, P_std, device)
    r = imf.logit_normal_dist(n, P_mean, P_std, device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    fm = torch.arange(n, device=device) >= n - int(n * data_proportion)
    return t, torch.where(fm, t, r), fm


MUTANTS = {
    "no stop-gradient on du/dt": ("compound", compound_no_stop_grad),
    "r tangent 1 instead of 0": ("compound", compound_r_tangent),
    "flow-matching samples at the end of the batch": ("sample_tr", sample_tr_suffix),
    "adaptive weight not detached": ("adp_wt_fn", lambda l, p, e: l / (l + e) ** p),
    "mean instead of sum over data dims": ("sq_err", lambda a, b, m: ((a - b) ** 2).flatten(1).mean(1)),
}

missed = []
for label, (name, fn) in MUTANTS.items():
    setattr(imf, name, fn)
    code = pytest.main(["-q", "-x", "-p", "no:cacheprovider", "tests/test_imf_reference.py", "-k", "objective"])
    setattr(imf, name, real[name])
    print(f"{'caught' if code != 0 else 'MISSED'}: {label}", flush=True)
    if code == 0:
        missed.append(label)
assert pytest.main(["-q", "-p", "no:cacheprovider", "tests/test_imf_reference.py", "-k", "objective"]) == 0
sys.exit(1 if missed else 0)
