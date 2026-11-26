"""Class-conditional generation: spiral arm labels, guided ambient iMF, which labels sampling
uses, guidance-scale lists, the arm-accuracy metric and the class-balanced ImageNet reference."""

import numpy as np
import pytest
import torch

from src.config import load
from src.data import Batch
from src.datasets import spiral
from src.metrics import spiral_scores
from src.models import MLP
from src.objectives import ambient_imf, imf
from src.training import draw, reference, samples

COND = ["model.num_classes=2", "objective.guidance=true"]
LOSS = dict(P_mean=-0.4, P_std=1.0, data_proportion=0.5, cfg_beta=1.0, class_dropout_prob=0.1,
            norm_p=0.0, norm_eps=0.01, guidance=True)


def mlp(num_classes=2, dim=2):
    torch.manual_seed(0)
    return MLP(dim=dim, hidden=32, depth=3, aux_head_depth=1, num_classes=num_classes)


def random_net(num_classes):
    """An MLP with random non-zero weights: at initialisation its output projection and residual
    gates are zero, so labels and guidance would not change its samples."""
    net = mlp(num_classes)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net


def test_spiral_labels_are_the_arms():
    """Arm 0 is the point reflection of arm 1, and the labels are drawn without disturbing the points."""
    x, arm = spiral.spiral(100_000, torch.Generator().manual_seed(0))
    assert set(arm.tolist()) == {0, 1} and abs(arm.float().mean().item() - 0.5) < 0.01
    t = 3 * torch.pi * torch.rand(100_000, generator=torch.Generator().manual_seed(0)).sqrt()  # the same first draw
    angle_on_arm1 = torch.atan2(x[:, 1], x[:, 0]) - torch.where(arm == 0, torch.pi, 0.0)
    assert (torch.cos(angle_on_arm1 - t) > 0.99).float().mean() > 0.999  # all but jitter crossing the origin
    batch = spiral.sample_clean(1000, torch.Generator().manual_seed(1))
    assert torch.equal(batch.cond, spiral.spiral(1000, torch.Generator().manual_seed(1))[1])


@pytest.mark.parametrize("seed", [1, 2])
def test_guided_ambient_loss_equals_guided_imf_on_clean_data(seed):
    batch = Batch(x=torch.randn(256, 2), sigma=torch.zeros(256), cond=torch.randint(0, 2, (256,)))
    torch.manual_seed(seed)
    ours, logs = ambient_imf.loss(mlp(), batch, t_margin=0.05, **LOSS)
    torch.manual_seed(seed)
    ref, _ = imf.loss(mlp(), batch, **LOSS)
    assert logs["valid"] == 1 and torch.allclose(ours, ref, rtol=1e-6)


def test_guided_ambient_loss_is_finite_with_masked_samples():
    net = mlp()
    batch = Batch(x=torch.randn(512, 2), sigma=torch.full((512,), 5.0), cond=torch.randint(0, 2, (512,)))
    torch.manual_seed(2)
    loss, logs = ambient_imf.loss(net, batch, t_margin=0.05, **LOSS)
    loss.backward()
    assert 0 < logs["valid"] < 0.5 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("config", ["configs/imf.yaml", "configs/ambient_imf.yaml"])
def test_unconditional_runs_sample_the_null_class(config):
    cfg = load(config, ["slug=test", "data.dim=2", "model.dim=2"])
    ref = reference(cfg)
    assert ref.cond is not None  # the spiral always carries its arms
    eps = torch.randn(ref.x.shape, generator=torch.Generator().manual_seed(cfg.seed))
    net = random_net(num_classes=0)
    out = samples(net, cfg, ref)
    assert list(out) == [1, 2] and torch.equal(out[2], draw(net, cfg, eps, 2, None, 1.0))


def test_conditional_runs_sample_the_reference_labels_at_each_guidance_scale():
    cfg = load("configs/imf.yaml", ["slug=test", "data.dim=2", "model.dim=2", *COND, "sampler.omega=[1.0, 2.5]"])
    ref, net = reference(cfg), random_net(num_classes=2)
    eps = torch.randn(ref.x.shape, generator=torch.Generator().manual_seed(cfg.seed))
    out = samples(net, cfg, ref)
    assert list(out) == ["1_w1", "1_w2.5", "2_w1", "2_w2.5"]
    assert torch.equal(out["2_w2.5"], draw(net, cfg, eps, 2, ref.cond, 2.5))
    assert not torch.allclose(out["2_w1"], draw(net, cfg, eps, 2, 1 - ref.cond, 1.0))  # labels matter
    assert not torch.allclose(out["2_w1"], out["2_w2.5"])  # so does the guidance scale
    single = load("configs/imf.yaml", ["slug=test", "data.dim=2", "model.dim=2", *COND])
    assert torch.equal(samples(net, single, ref)[2], out["2_w1"])


def test_arm_accuracy_counts_samples_on_the_requested_arm():
    cfg = load("configs/imf.yaml", ["slug=test", *COND, "metric.n_samples=1000"])
    ref = reference(cfg)
    right, swapped = spiral_scores(cfg, {1: ref.x}, ref), spiral_scores(cfg, {1: -ref.x}, ref)
    assert right["arm_accuracy@1"] == 1.0 and swapped["arm_accuracy@1"] < 0.05
    unconditional = load("configs/imf.yaml", ["slug=test", "metric.n_samples=1000"])
    assert "arm_accuracy@1" not in spiral_scores(unconditional, {1: ref.x}, ref)


def test_imagenet_reference_is_class_balanced_and_seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("NRFM_DATA", str(tmp_path))
    (tmp_path / "imagenet").mkdir()
    np.save(tmp_path / "imagenet" / "val_latents.npy", np.zeros((40, 4, 32, 32), dtype=np.float32))
    np.save(tmp_path / "imagenet" / "val_labels.npy", np.arange(40) % 4)
    cfg = load("configs/imagenet.yaml", ["slug=test", "model.num_classes=4", "metric.n_samples=12"])
    ref = reference(cfg)
    assert torch.equal(ref.cond.bincount(), torch.full((4,), 3)) and torch.equal(ref.cond, reference(cfg).cond)
    with pytest.raises(AssertionError):
        reference(load("configs/imagenet.yaml", ["slug=test", "model.num_classes=4", "metric.n_samples=10"]))
    with pytest.raises(AssertionError):
        reference(load("configs/imagenet.yaml", ["slug=test", "model.num_classes=4", "metric.n_samples=48"]))
