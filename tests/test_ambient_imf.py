"""The ambient target against closed-form answers for x ~ N(0, A^2) in one dimension:
the clean path has z_t ~ N(0, s_t^2) with s_t^2 = (1 - t)^2 A^2 + t^2 and velocity
E[e - x | z_t] = z_t (t - (1 - t) A^2) / s_t^2."""

import pytest
import torch

from src.data import Batch
from src.models import MLP
from src.objectives import ambient_imf, imf
from src.path import interpolate

A, N = 0.5, 2_000_000
IMF = dict(P_mean=-0.4, P_std=1.0, data_proportion=0.5, norm_p=1.0, norm_eps=0.01)


def noisy_gaussian(sigma, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = A * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y = x + sigma * torch.randn(N, 1, generator=g, dtype=torch.float64)
    e = torch.randn(N, 1, generator=g, dtype=torch.float64)
    return ambient_imf.ambient_noising(y, torch.full((N,), sigma, dtype=torch.float64),
                                       torch.full((N,), t, dtype=torch.float64), e)


@pytest.mark.parametrize("sigma, t", [(0.3, 0.5), (1.0, 0.7)])
def test_noisy_path_has_the_clean_marginal(sigma, t):
    z, _ = noisy_gaussian(sigma, t)
    assert abs(z.var().item() / ((1 - t) ** 2 * A ** 2 + t ** 2) - 1) < 0.01


@pytest.mark.parametrize("sigma, t", [(0.3, 0.5), (1.0, 0.7)])
def test_target_is_unbiased_for_the_velocity(sigma, t):
    """E[v_hat | z] is linear in z here, so its regression slope must match the exact one."""
    z, v_hat = noisy_gaussian(sigma, t)
    slope = ((v_hat * z).mean() / (z * z).mean()).item()
    exact = (t - (1 - t) * A ** 2) / ((1 - t) ** 2 * A ** 2 + t ** 2)
    assert abs(slope - exact) < 0.02 * abs(exact), (slope, exact)


def test_clean_samples_reduce_to_imf():
    torch.manual_seed(0)
    x, e, t = torch.randn(64, 2), torch.randn(64, 2), torch.rand(64)
    z, v_hat = ambient_imf.ambient_noising(x, torch.zeros(64), t, e)
    assert torch.equal(z, interpolate(x, e, t)) and torch.equal(v_hat, e - x)


def mlp():
    torch.manual_seed(0)
    return MLP(dim=2, hidden=32, depth=3, aux_head_depth=1, num_classes=0)


def test_loss_equals_imf_on_clean_data():
    batch = Batch(x=torch.randn(256, 2), sigma=torch.zeros(256))
    torch.manual_seed(1)
    ours, logs = ambient_imf.loss(mlp(), batch, t_margin=0.05, **IMF)
    torch.manual_seed(1)
    ref, _ = imf.loss(mlp(), batch, cfg_beta=1.0, class_dropout_prob=0.1, guidance=False, **IMF)
    assert logs["valid"] == 1 and torch.allclose(ours, ref, rtol=1e-4)


def test_masked_samples_keep_loss_and_gradients_finite():
    net = mlp()
    batch = Batch(x=torch.randn(512, 2), sigma=torch.full((512,), 5.0))  # t_n = 5/6: mostly masked
    torch.manual_seed(2)
    loss, logs = ambient_imf.loss(net, batch, t_margin=0.05, **IMF)
    loss.backward()
    assert 0 < logs["valid"] < 0.5 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
