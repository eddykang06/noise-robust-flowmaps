"""End-to-end training on the GPU, in a temporary directory.

These check the pipeline, not iMF's hyperparameters: they shorten training and
use lr 1e-3 and EMA 0.99 so a few thousand steps suffice.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from src.config import load
from src.metrics import energy_distance
from src.training import evaluate, load_run, reference, train

pytestmark = pytest.mark.gpu
FAST = ["train.steps=4000", "train.eval_every=2000", "train.lr=1.0e-3", "train.ema=0.99"]


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


def run(tmp: Path, name: str, extra=()):
    cfg = load("configs/toy.yaml", ["slug=test", f"out_root={tmp}/{name}", f"runs_dir={tmp}",
                                    *FAST, *extra])
    train(cfg)
    return cfg


def metrics(tmp: Path, name: str) -> list[dict]:
    """The metrics of every record written by run `name`, in order."""
    records = map(json.loads, (tmp / "test.jsonl").read_text().splitlines())
    return [r["metrics"] for r in records if r["config"]["out_root"] == f"{tmp}/{name}"]


@pytest.fixture(scope="module")
def reference_run(tmp):
    run(tmp, "a")
    return metrics(tmp, "a")


def test_identical_runs_are_bitwise_reproducible(tmp, reference_run):
    run(tmp, "b")
    assert len(reference_run) == 2 and metrics(tmp, "b") == reference_run


def test_learns_the_checkerboard_in_one_step(reference_run):
    ref = reference(load("configs/toy.yaml", ["slug=test"])).x
    prior = energy_distance(torch.randn(ref.shape, generator=torch.Generator().manual_seed(6)), ref)
    assert reference_run[-1]["energy_distance@1"] < 0.3 * prior, (reference_run[-1], prior)


def test_interrupted_run_resumes_bitwise(tmp, reference_run, monkeypatch):
    real_save = torch.save

    def save_then_interrupt(obj, path):
        real_save(obj, path)
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(tmp, "resume")
    monkeypatch.undo()
    cfg = run(tmp, "resume")
    assert metrics(tmp, "resume")[-1] == reference_run[-1]

    _, net = load_run(Path(cfg.out_root) / cfg.slug / cfg.run_id)
    assert evaluate(net, cfg, reference(cfg).to("cuda")) == reference_run[-1]


@pytest.mark.parametrize("name, extra", [
    ("corrupted", ["corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"]),
    ("clean-only", ["corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}", "data.keep_noisy=false"]),
    ("bf16", ["train.precision=bf16"]),
])
def test_arms_run(tmp, name, extra):
    run(tmp, name, ["train.steps=300", "train.eval_every=300", *extra])
    assert all(math.isfinite(m) for m in metrics(tmp, name)[-1].values())
