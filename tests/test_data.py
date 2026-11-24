from dataclasses import replace

import pytest
import torch

from src.config import load
from src.corrupt import box_uniform_var, gaussian, gaussian_uniform_var, uniform
from src.data import Batch
from src.embed import basis, frame
from src.metrics import wasserstein2
from src.toy import sample_clean, spiral
from src.training import build_dataset, reference

CORRUPT = ["slug=test", "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"]


def test_batch_indexes_set_fields_and_keeps_unset_ones():
    b = Batch(x=torch.arange(6.0).view(3, 2), sigma=torch.tensor([0.0, 1.0, 2.0]), cond=torch.arange(3))
    sub = b[torch.tensor([2, 0])]
    assert torch.equal(sub.x, b.x[[2, 0]]) and torch.equal(sub.cond, torch.tensor([2, 0]))
    assert sub.mask is None and len(sub) == 2


def test_spiral_has_unit_variance(gen):
    var = spiral(200_000, gen(0)).var(0)
    assert torch.allclose(var, torch.ones(2), atol=0.1), var


def test_dataset_is_fixed_by_seed_and_distinct_from_reference():
    cfg = load("configs/imf.yaml", CORRUPT)
    d = build_dataset(cfg)
    assert torch.equal(d.x, build_dataset(cfg).x)
    assert not torch.equal(d.x, build_dataset(replace(cfg, seed=43)).x)
    assert not torch.equal(reference(cfg).x[:100], d.x[:100])


def test_clean_only_keeps_exactly_the_clean_samples():
    cfg = load("configs/imf.yaml", CORRUPT)
    d = build_dataset(cfg)
    clean = build_dataset(replace(cfg, data={**cfg.data, "keep_noisy": False}))
    assert torch.equal(clean.x, d.x[d.sigma == 0]) and (clean.sigma == 0).all()


@pytest.mark.parametrize("corrupt", [gaussian, uniform])
def test_fixed_sigma_corruptions(gen, corrupt):
    clean = sample_clean("spiral", 200_000, gen(1))
    obs = corrupt(clean, gen(2), sigma=0.3, clean_frac=0.1)
    noisy = obs.sigma > 0
    noise = (obs.x - clean.x)[noisy]
    assert abs(noisy.float().mean() - 0.9) < 0.005
    assert abs(noise.std() - 0.3) < 0.003 and abs(noise.mean()) < 0.003
    assert torch.equal(obs.sigma.unique(), torch.tensor([0.0, 0.3]))
    assert torch.equal(obs.x[~noisy], clean.x[~noisy])
    if corrupt is uniform:
        assert noise.abs().max() <= 0.3 * 3 ** 0.5 + 1e-5


@pytest.mark.parametrize("corrupt", [gaussian_uniform_var, box_uniform_var])
def test_per_sample_variance_corruptions(gen, corrupt):
    clean = sample_clean("spiral", 200_000, gen(1))
    obs = corrupt(clean, gen(2), var_max=10.0, clean_frac=1 / 3)
    noisy = obs.sigma > 0
    var = obs.sigma[noisy] ** 2
    assert abs(noisy.float().mean() - 2 / 3) < 0.005
    assert var.max() <= 10.0 and abs(var.mean() - 5.0) < 0.05 and abs(var.std() - 10 / 12 ** 0.5) < 0.05
    z = (obs.x - clean.x)[noisy] / obs.sigma[noisy, None]
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01
    assert torch.equal(obs.x[~noisy], clean.x[~noisy])
    if corrupt is box_uniform_var:
        assert z.abs().max() <= 3 ** 0.5 + 1e-5


def test_wasserstein2(gen):
    p, q = torch.randn(1000, 2, generator=gen(4)), torch.randn(1000, 2, generator=gen(5))
    assert wasserstein2(p, p) < 1e-6
    assert abs(wasserstein2(p, q) - wasserstein2(q, p)) < 1e-9
    # A translation is optimally coupled by itself, so W2 is its length.
    assert abs(wasserstein2(p, p + torch.tensor([3.0, 4.0])) - 5.0) < 1e-5
    # In one dimension the optimal plan matches sorted samples.
    a, b = p[:, :1].double(), q[:, :1].double()
    sorted_w2 = (a.sort(0).values - b.sort(0).values).square().mean().sqrt().item()
    assert abs(wasserstein2(a, b) - sorted_w2) < 1e-9


def test_full_wasserstein2_splits_into_plane_and_off_plane(gen):
    """Against data on the plane, the off-plane cost of a pair does not depend on the
    matching, so W2 in D dimensions is sqrt(W2 in the plane^2 + mean squared distance off it)."""
    dim, n = 8, 256
    ref = torch.cat([sample_clean("spiral", n, gen(0)).x, torch.zeros(n, dim - 2)], 1)
    x = torch.randn(n, dim, generator=gen(1))
    full = wasserstein2(x @ basis(dim).T, ref @ basis(dim).T)
    split = (wasserstein2(x[:, :2], ref[:, :2]) ** 2 + x[:, 2:].square().sum(1).mean().item()) ** 0.5
    assert abs(full - split) < 1e-4


@pytest.mark.parametrize("dim", [2, 8, 512])
def test_basis_is_fixed_and_orthogonal(dim):
    q = basis(dim)
    assert torch.equal(q, basis(dim))
    assert torch.allclose(q.T @ q, torch.eye(dim), atol=1e-5)


def test_embedded_dataset_lies_on_the_plane_and_noise_keeps_its_shape():
    """Clean samples are the 2-D toy in the plane, with nothing off it; uniform
    noise is added in the frame, so within the plane it stays inside its box."""
    var = ["data.dim=16", "model.dim=16"]
    clean = build_dataset(load("configs/imf.yaml", ["slug=test", *var]))
    gen = torch.Generator().manual_seed(42)
    assert torch.allclose(frame(clean.x)[:, :2], sample_clean("spiral", len(clean), gen).x, atol=1e-5)
    assert frame(clean.x)[:, 2:].abs().max() < 1e-5

    box = ["corruption={name: box_uniform_var, var_max: 1.0, clean_frac: 0.3333}"]
    obs = build_dataset(load("configs/imf.yaml", ["slug=test", *var, *box]))
    noise = frame(obs.x - clean.x)[obs.sigma > 0] / obs.sigma[obs.sigma > 0, None]
    assert noise.abs().max() <= 3 ** 0.5 + 1e-3
    assert abs(noise.std() - 1.0) < 0.02
