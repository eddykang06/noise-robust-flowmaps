"""The iMF objective and sampler against closed-form answers.

For data x ~ N(0, a^2 I) on z_t = (1 - t) x + t e, z_t ~ N(0, s_t^2 I) with
s_t^2 = (1 - t)^2 a^2 + t^2 and the flow is z_r = (s_r / s_t) z_t, so
u(z, r, t) = z (1 - s_r / s_t) / (t - r) and v(z, t) = z (t - (1 - t) a^2) / s_t^2.
"""

import pytest
import torch

from src.data import Batch
from src.models import MLP
from src.objectives import imf
from src.sampling import imf as sample

A = 0.5


class GaussianFlow(torch.nn.Module):
    """The exact (u, v) for x ~ N(0, A^2 I), with the iMF network signature."""

    num_classes = 0

    def s(self, t):
        return ((1 - t) ** 2 * A ** 2 + t ** 2).sqrt()

    def forward(self, z, t, h, omega, t_min, t_max, y):
        r = t - h
        vel = (t - (1 - t) * A ** 2) / self.s(t) ** 2
        avg = (1 - self.s(r) / self.s(t)) / torch.where(h == 0, torch.ones_like(h), h)
        return z * torch.where(h == 0, vel, avg)[:, None], z * vel[:, None]


def random_mlp(num_classes=0):
    """A toy iMF network with every parameter random, so neither head is zero."""
    torch.manual_seed(0)
    net = MLP(dim=2, hidden=64, depth=3, aux_head_depth=1, num_classes=num_classes)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return net


def guide_inputs(n, omega):
    return torch.full((n,), omega), torch.zeros(n), torch.ones(n)


def test_sample_tr_orders_times_and_sets_flow_matching_prefix():
    t, r, fm = imf.sample_tr(1000, -0.4, 1.0, 0.5, "cpu")
    assert (t >= r).all() and ((0 < r) & (t < 1)).all()
    assert torch.equal(fm, torch.arange(1000) < 500)
    assert torch.equal(t[fm], r[fm]) and (t[~fm] > r[~fm]).all()


@pytest.mark.parametrize("beta, median", [(1.0, 8 ** 0.5), (2.0, 1 / (1 - 0.5 * 7 / 8))])
def test_cfg_scale_has_power_law_density_on_1_to_8(beta, median):
    torch.manual_seed(0)
    s = imf.sample_cfg_scale(200_000, beta, "cpu")
    assert s.min() >= 1 and s.max() <= 8
    assert abs(s.median() / median - 1) < 0.01


def test_cfg_interval_ranges_and_flow_matching_override():
    fm = torch.arange(1000) < 300
    t_min, t_max = imf.sample_cfg_interval(1000, fm, "cpu")
    assert (t_min[fm] == 0).all() and (t_max[fm] == 1).all()
    assert ((0 <= t_min) & (t_min <= 0.5) & (0.5 <= t_max) & (t_max <= 1)).all()


def test_cond_drop_nulls_a_prefix_and_restores_the_unguided_target():
    torch.manual_seed(0)
    net, n = GaussianFlow(), 10_000
    net.num_classes = 7
    v_t, v_g = torch.zeros(n, 2), torch.ones(n, 2)
    y, v = imf.cond_drop(net, v_t, v_g, torch.zeros(n, dtype=torch.long), 0.1)
    k = int((y == 7).sum())
    assert abs(k / n - 0.1) < 0.01 and (y[:k] == 7).all() and (y[k:] == 0).all()
    assert (v[:k] == 0).all() and (v[k:] == 1).all()


def test_compound_prediction_is_the_exact_velocity():
    """With the exact flow and its own velocity as tangent, V = u + (t - r) du/dt = v."""
    net, n = GaussianFlow(), 4096
    torch.manual_seed(0)
    z = torch.randn(n, 2, dtype=torch.float64)
    a, b = torch.rand(n, dtype=torch.float64), torch.rand(n, dtype=torch.float64)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    r[: n // 4] = t[: n // 4]
    omega, t_min, t_max = (x.double() for x in guide_inputs(n, 1.0))
    y = torch.zeros(n, dtype=torch.long)
    v_exact = net(z, t, torch.zeros_like(t), omega, t_min, t_max, y)[1]
    V, v = imf.compound(net, z, t, r, omega, t_min, t_max, y, v_exact)
    assert torch.allclose(V, v_exact, atol=1e-10)
    assert torch.equal(v, v_exact)


def test_jvp_is_the_total_time_derivative():
    net, n = random_mlp(), 512
    torch.manual_seed(1)
    z, tangent = torch.randn(n, 2), torch.randn(n, 2)
    t = 0.2 + 0.6 * torch.rand(n)
    r = t * torch.rand(n)
    omega, t_min, t_max = guide_inputs(n, 1.0)
    y = torch.zeros(n, dtype=torch.long)
    u = lambda z, t: net(z, t, t - r, omega, t_min, t_max, y)[0]
    V, _ = imf.compound(net, z, t, r, omega, t_min, t_max, y, tangent)
    du_dt = (V - u(z, t)) / (t - r)[:, None]
    d = 1e-2
    fd = (u(z + d * tangent, t + d) - u(z - d * tangent, t - d)) / (2 * d)
    assert torch.allclose(du_dt, fd, rtol=1e-2, atol=1e-3), (du_dt - fd).abs().max()


def test_guidance_reduces_to_the_unguided_target():
    net, n = random_mlp(num_classes=4), 1000
    torch.manual_seed(2)
    z, v_t, t = torch.randn(n, 2), torch.randn(n, 2), torch.rand(n)
    y = torch.randint(0, 4, (n,))
    fm = torch.arange(n) < n // 2
    _, t_min, t_max = guide_inputs(n, 1.0)
    v_g, _ = imf.guidance_fn(net, v_t, z, t, y, fm, torch.ones(n), t_min, t_max)
    assert torch.equal(v_g, v_t)  # omega = 1

    omega = torch.full((n,), 3.0)
    t_min, t_max = torch.full((n,), 0.4), torch.full((n,), 0.6)
    v_g, _ = imf.guidance_fn(net, v_t, z, t, y, fm, omega, t_min, t_max)
    outside = ~fm & ((t < 0.4) | (t > 0.6))
    assert torch.equal(v_g[outside], v_t[outside])  # gated to omega = 1 outside the interval
    assert (v_g[fm] != v_t[fm]).any()  # flow-matching samples are always guided


@pytest.mark.parametrize("guidance", [False, True])
def test_loss_trains_both_heads(guidance):
    net = random_mlp(num_classes=4)
    torch.manual_seed(3)
    batch = Batch(x=torch.randn(256, 2), sigma=torch.zeros(256), cond=torch.randint(0, 4, (256,)))
    loss, logs = imf.loss(net, batch, P_mean=-0.4, P_std=1.0, data_proportion=0.5, cfg_beta=1.0,
                          class_dropout_prob=0.1, norm_p=1.0, norm_eps=0.01, guidance=guidance)
    loss.backward()
    assert torch.isfinite(loss) and set(logs) == {"loss_u", "loss_v"}
    for head in (net.u_head, net.v_head):
        assert head.out._flax_linear.weight.grad.abs().sum() > 0


def test_adaptive_weight():
    loss = torch.tensor([0.0, 1.0, 9.99], requires_grad=True)
    w = imf.adp_wt_fn(loss, norm_p=1.0, norm_eps=0.01)
    assert torch.allclose(w, loss / (loss + 0.01))
    w.sum().backward()
    assert torch.allclose(loss.grad, 1 / (loss.detach() + 0.01))  # the weight is stop-gradient


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_sampler_is_exact_with_the_exact_flow(steps):
    torch.manual_seed(4)
    eps = torch.randn(4096, 2, dtype=torch.float64)
    x = sample(GaussianFlow(), eps, steps, None, omega=1.0, t_min=0.0, t_max=1.0)
    assert torch.allclose(x, A * eps, atol=1e-12)
