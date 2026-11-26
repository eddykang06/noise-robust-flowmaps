"""MNIST and ImageNet loading, corruption of image-shaped data, and the image metrics, on small
synthetic files under a temporary $NRFM_DATA (no downloads)."""

import numpy as np
import pytest
import torch

from src.config import load
from src.corrupt import gaussian
from src.data import Batch
from src.datasets import imagenet, mnist
from src.metrics import class_score, frechet, precision_recall
from src.models import MODELS
from src.training import build_dataset, draw, reference, train


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NRFM_DATA", str(tmp_path))
    rng = np.random.default_rng(0)
    (tmp_path / "mnist").mkdir()
    for split, n in [("train", 200), ("test", 300)]:
        np.savez(tmp_path / "mnist" / f"{split}.npz", images=rng.integers(0, 256, (n, 28, 28), dtype=np.uint8),
                 labels=rng.integers(0, 10, n, dtype=np.uint8))
    (tmp_path / "imagenet").mkdir()
    for split, n in [("train", 40), ("val", 20)]:
        np.save(tmp_path / "imagenet" / f"{split}_latents.npy", rng.standard_normal((n, 4, 32, 32), dtype=np.float32))
        np.save(tmp_path / "imagenet" / f"{split}_labels.npy", rng.integers(0, 1000, n))
    return tmp_path


def test_mnist_digits_are_padded_scaled_and_keep_their_labels(data_root):
    d = build_dataset(load("configs/mnist.yaml", ["slug=test", "data.n=100"]))
    files = np.load(data_root / "mnist" / "train.npz")
    idx = torch.randperm(200, generator=torch.Generator().manual_seed(42))[:100]
    assert d.x.shape == (100, 1, 32, 32) and (d.sigma == 0).all()
    assert torch.equal(d.cond, torch.from_numpy(files["labels"]).long()[idx])
    assert torch.equal(mnist.to_images(d.x)[:, 0, 2:30, 2:30], torch.from_numpy(files["images"])[idx])
    assert (d.x[..., :2, :] == -1).all() and d.x.min() >= -1 and d.x.max() <= 1


def test_mnist_reference_is_the_seeded_test_split(data_root):
    cfg = load("configs/mnist.yaml", ["slug=test", "metric.n_samples=30"])
    ref = reference(cfg)
    labels = torch.from_numpy(np.load(data_root / "mnist" / "test.npz")["labels"]).long()
    assert len(ref) == 30 and torch.equal(ref.cond, reference(cfg).cond)
    assert torch.equal(ref.cond, labels[torch.randperm(300, generator=torch.Generator().manual_seed(10_042))[:30]])


def test_corruption_acts_per_sample_on_images(gen):
    clean = Batch(x=torch.zeros(20_000, 1, 8, 8), sigma=torch.zeros(20_000), cond=torch.arange(20_000))
    obs = gaussian(clean, gen(0), sigma=0.5, clean_frac=0.25)
    noisy = obs.sigma > 0
    assert obs.x.shape == clean.x.shape and torch.equal(obs.cond, clean.cond)
    assert abs(noisy.float().mean() - 0.75) < 0.01 and (obs.x[~noisy] == 0).all()
    assert abs(obs.x[noisy].std() - 0.5) < 0.005


def test_imagenet_training_set_is_memory_mapped_and_its_corruption_cached(data_root, monkeypatch):
    clean = build_dataset(load("configs/imagenet.yaml", ["slug=test"]))
    assert clean.x.shape == (40, 4, 32, 32) and clean.cond.dtype == torch.int64
    assert torch.equal(clean.x, torch.from_numpy(np.load(data_root / "imagenet" / "train_latents.npy")))

    monkeypatch.setattr(imagenet, "CHUNK", 7)  # several chunks, one generator
    cfg = load("configs/imagenet.yaml", ["slug=test", "corruption={name: gaussian, sigma: 0.5, clean_frac: 0.5}"])
    first, cached = build_dataset(cfg), build_dataset(cfg)
    assert len(list((data_root / "imagenet" / "observed").iterdir())) == 1
    assert torch.equal(first.x, cached.x) and torch.equal(first.sigma, cached.sigma)
    assert torch.equal(first.x[first.sigma == 0], clean.x[first.sigma == 0])
    assert set(first.sigma.tolist()) == {0.0, 0.5} and torch.equal(first.cond, clean.cond)


def test_sampling_in_batches_matches_one_pass(gen):
    torch.manual_seed(0)
    net = MODELS["mlp"](dim=2, hidden=32, depth=2, aux_head_depth=1, num_classes=0)
    eps = torch.randn(50, 2, generator=gen(1))
    whole, batched = (draw(net, load("configs/imf.yaml", ["slug=test", *extra]), eps, 2, None, 1.0)
                      for extra in ([], ["sampler.batch=7"]))
    assert torch.allclose(whole, batched, atol=1e-6)


def test_frechet_distance(gen):
    a, b = torch.randn(20_000, 3, generator=gen(0)), torch.randn(20_000, 3, generator=gen(1))
    assert abs(frechet(a, a)) < 1e-8
    assert abs(frechet(a, b + torch.tensor([3.0, 0.0, 4.0])) - 25.0) < 0.2


def test_class_score_counts_confident_distinct_classes():
    assert abs(class_score(torch.full((100, 10), 0.1)) - 1.0) < 1e-6
    assert abs(class_score(torch.eye(10).repeat(10, 1)) - 10.0) < 1e-4


def test_precision_recall(gen):
    real = torch.randn(2000, 4, generator=gen(0))
    assert precision_recall(real, real) == (1.0, 1.0)
    precision, recall = precision_recall(real, real + 100)
    assert precision == 0.0 and recall == 0.0


@pytest.mark.gpu
def test_mnist_run_trains_and_scores(data_root, tmp_path):
    torch.save(mnist.Classifier().state_dict(), data_root / "mnist" / "classifier.pt")
    mnist.classifier.cache_clear()
    small = ["model.depth=2", "model.aux_head_depth=1", "model.hidden_size=64", "model.num_heads=4"]
    cfg = load("configs/mnist.yaml", ["slug=test", f"out_root={tmp_path}/runs", f"runs_dir={tmp_path}", *small,
                                      "train.steps=20", "train.eval_every=20", "metric.n_samples=200",
                                      "corruption={name: gaussian, sigma: 0.5, clean_frac: 0.5}"])
    train(cfg)
    run_dir = tmp_path / "runs" / "test" / cfg.run_id
    record = (tmp_path / "test.jsonl").read_text().splitlines()[-1]
    assert (run_dir / "true-vs-generated.png").stat().st_size > 0
    for key in ["fid@1", "precision@2", "recall@1", "is@2", "accuracy@1"]:
        assert f'"{key}"' in record
