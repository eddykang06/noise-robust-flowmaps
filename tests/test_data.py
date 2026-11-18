from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from src.config import load
from src.corrupt import gaussian
from src.data import Batch, sample_clean
from src.embed import basis, split
from src.metrics import energy_distance
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


def test_energy_distance(gen):
    p, q = torch.randn(5000, 2, generator=gen(4)), torch.randn(5000, 2, generator=gen(5))
    assert abs(energy_distance(p, p)) < 1e-6
    assert abs(energy_distance(p, q) - energy_distance(q, p)) < 1e-6
    assert energy_distance(p, q) < 0.01 < 0.3 < energy_distance(p, q + 1)


def test_unset_dim_is_the_unembedded_2d_pipeline(gen):
    """Without `data.dim`, the dataset is the toy sample corrupted in place, as before
    embedding existed, and `dim: 2` gives the same data."""
    cfg = load("configs/toy.yaml", CORRUPT)
    g = gen(cfg.seed)
    expected = gaussian(sample_clean("checkerboard", cfg.data["n"], g), g, sigma=0.3, clean_frac=0.1)
    d = build_dataset(cfg)
    assert torch.equal(d.x, expected.x) and torch.equal(d.sigma, expected.sigma)
    two = load("configs/toy.yaml", [*CORRUPT, "data.dim=2", "model.dim=2"])
    assert torch.equal(build_dataset(two).x, d.x) and torch.equal(reference(two).x, reference(cfg).x)
    plane, rest = split(d.x, 2)
    assert plane is d.x and rest is None


@pytest.mark.parametrize("dim", [3, 16, 512])
def test_basis_is_fixed_and_orthogonal(dim):
    q = basis(dim)
    assert torch.equal(q, basis(dim))
    assert torch.allclose(q.T @ q, torch.eye(dim), atol=1e-5)


def test_embedded_toy_lies_on_the_plane_and_is_corrupted_in_the_frame(gen):
    """Clean samples are the 2-D toy in the plane with nothing off it; corrupted ones
    are, in Q's frame, the zero-padded toy corrupted coordinate by coordinate."""
    over = ["slug=test", "data.dim=16", "model.dim=16"]
    clean = build_dataset(load("configs/toy.yaml", over))
    plane, rest = split(clean.x, 16)
    assert clean.x.shape == (10_000, 16)
    assert torch.allclose(plane, sample_clean("checkerboard", 10_000, gen(42)).x, atol=1e-5)
    assert rest.abs().max() < 1e-5

    obs = build_dataset(load("configs/toy.yaml", [*over, CORRUPT[1]]))
    g = gen(42)
    toy = sample_clean("checkerboard", 10_000, g)
    expected = gaussian(replace(toy, x=F.pad(toy.x, (0, 14))), g, sigma=0.3, clean_frac=0.1)
    assert torch.allclose(torch.cat(split(obs.x, 16), 1), expected.x, atol=1e-5)
    assert torch.equal(obs.sigma, expected.sigma)
