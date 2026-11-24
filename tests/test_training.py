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
from src.metrics import wasserstein2
from src.training import evaluate, load_run, reference, train

pytestmark = pytest.mark.gpu
FAST = ["train.steps=4000", "train.eval_every=2000", "train.lr=1.0e-3", "train.ema=0.99"]


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


def run(tmp: Path, name: str, extra=(), config="configs/imf.yaml"):
    cfg = load(config, ["slug=test", f"out_root={tmp}/{name}", f"runs_dir={tmp}", *FAST, *extra])
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


def test_learns_the_spiral_in_one_step(reference_run):
    """One step moves the prior most of the way onto the spiral, in all dimensions."""
    ref = reference(load("configs/imf.yaml", ["slug=test"])).x
    prior = wasserstein2(torch.randn(ref.shape, generator=torch.Generator().manual_seed(6)), ref)
    assert reference_run[-1]["wasserstein2_full@1"] < 0.5 * prior, (reference_run[-1], prior)


def test_toy_run_writes_true_vs_generated_figure(tmp, reference_run):
    cfg = load("configs/imf.yaml", ["slug=test", f"out_root={tmp}/a", f"runs_dir={tmp}", *FAST])
    assert (Path(cfg.out_root) / cfg.slug / cfg.run_id / "true-vs-generated.png").stat().st_size > 0


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


GAUSSIAN = "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.75}"
UNIFORM = "corruption={name: uniform, sigma: 0.3, clean_frac: 0.75}"


@pytest.mark.parametrize("name, extra, config", [
    ("gaussian", [GAUSSIAN], "configs/imf.yaml"),
    ("clean-only", [GAUSSIAN, "data.keep_noisy=false"], "configs/imf.yaml"),
    ("ambient-gaussian", [GAUSSIAN], "configs/ambient_imf.yaml"),
    ("ambient-uniform", [UNIFORM], "configs/ambient_imf.yaml"),
    ("bf16", ["train.precision=bf16"], "configs/imf.yaml"),
    ("2-D", ["data.dim=2", "model.dim=2"], "configs/imf.yaml"),
])
def test_arms_run(tmp, name, extra, config):
    run(tmp, name, ["train.steps=300", "train.eval_every=300", *extra], config)
    last = metrics(tmp, name)[-1]
    assert all(math.isfinite(m) for m in last.values())
    assert ("off_plane_rms@1" in last) == ("wasserstein2_full@1" in last) == (name != "2-D")
