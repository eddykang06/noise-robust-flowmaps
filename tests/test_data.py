from dataclasses import replace

import pytest
import torch

from src.config import load
from src.corrupt import gaussian
from src.data import Batch, sample_clean
from src.metrics import wasserstein2
from src.toy import TOYS
from src.training import build_dataset, reference

CORRUPT = ["slug=test", "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"]


def test_batch_indexes_set_fields_and_keeps_unset_ones():
    b = Batch(x=torch.arange(6.0).view(3, 2), sigma=torch.tensor([0.0, 1.0, 2.0]), cond=torch.arange(3))
    sub = b[torch.tensor([2, 0])]
    assert torch.equal(sub.x, b.x[[2, 0]]) and torch.equal(sub.cond, torch.tensor([2, 0]))
    assert sub.mask is None and len(sub) == 2


@pytest.mark.parametrize("name", TOYS)
def test_toys_have_unit_variance(name, gen):
    var = TOYS[name](200_000, gen(0)).var(0)
    assert torch.allclose(var, torch.ones(2), atol=0.1), var


def test_checkerboard_mass_only_on_dark_squares(gen):
    xy = (TOYS["checkerboard"](200_000, gen(0)) * 1.15).floor()
    assert ((xy[:, 0] + xy[:, 1]) % 2 == 0).all()


def test_dataset_is_fixed_by_seed_and_distinct_from_reference():
    cfg = load("configs/toy.yaml", CORRUPT)
    d = build_dataset(cfg)
    assert torch.equal(d.x, build_dataset(cfg).x)
    assert not torch.equal(d.x, build_dataset(replace(cfg, seed=43)).x)
    assert not torch.equal(reference(cfg).x[:100], d.x[:100])


def test_clean_only_keeps_exactly_the_clean_samples():
    cfg = load("configs/toy.yaml", CORRUPT)
    d = build_dataset(cfg)
    clean = build_dataset(replace(cfg, data={**cfg.data, "keep_noisy": False}))
    assert torch.equal(clean.x, d.x[d.sigma == 0]) and (clean.sigma == 0).all()


def test_gaussian_corruption(gen):
    clean = sample_clean("checkerboard", 200_000, gen(1))
    obs = gaussian(clean, gen(2), sigma=0.3, clean_frac=0.1)
    noisy = obs.sigma > 0
    assert abs(noisy.float().mean() - 0.9) < 0.005
    assert abs((obs.x - clean.x)[noisy].std() - 0.3) < 0.003
    assert torch.equal(obs.sigma.unique(), torch.tensor([0.0, 0.3]))
    assert torch.equal(obs.x[~noisy], clean.x[~noisy])


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
