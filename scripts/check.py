"""Correctness and reproducibility checks for the shared pipeline. GPU job only:

    python -m scripts.check

Unit checks compare against Gaussian data x ~ N(0, a^2 I), whose flow is known
in closed form. End-to-end checks train tiny runs in a temporary directory and
never write to research/runs/. Every check runs; the exit code is non-zero if
any failed.
"""

import json
import math
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

from src import training
from src.config import load
from src.corrupt import gaussian
from src.data import Batch, sample_clean
from src.metrics import energy_distance
from src.models import MLP
from src.objectives import flow_matching
from src.objectives.meanflow import target
from src.path import bcast, interpolate
from src.sampling import SAMPLERS
from src.toy import TOYS


START, FAILED = time.time(), []


def check(name: str, ok: bool, detail=""):
    print(f"[{time.time() - START:6.1f}s] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def section(fn, *args):
    """Run one group of checks; an exception fails the group without stopping the rest."""
    try:
        fn(*args)
    except Exception:
        traceback.print_exc()
        check(f"{fn.__name__} raised", False)


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class GaussianFlow(torch.nn.Module):
    """Exact average velocity for x ~ N(0, a^2 I) on z_t = (1 - t) x + t eps.

    z_t ~ N(0, s_t^2 I) with s_t^2 = (1 - t)^2 a^2 + t^2, and the flow is the
    scaling z_r = (s_r / s_t) z_t, so u(z, r, t) = z (1 - s_r / s_t) / (t - r);
    at r = t it is the velocity v(z, t) = z (t - (1 - t) a^2) / s_t^2.
    """

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def s(self, t):
        return ((1 - t) ** 2 * self.a ** 2 + t ** 2).sqrt()

    def forward(self, z, r, t, cond=None):
        same = t == r
        avg = (1 - self.s(r) / self.s(t)) / torch.where(same, torch.ones_like(t), t - r)
        vel = (t - (1 - t) * self.a ** 2) / self.s(t) ** 2
        return z * bcast(torch.where(same, vel, avg), z)


def check_config():
    args = ["slug=check", "objective.ratio=0.5", "sampler.steps=[1,4]"]
    cfg = load("configs/toy.yaml", args)
    check("config overrides", cfg.objective["ratio"] == 0.5 and cfg.sampler["steps"] == [1, 4])
    check("run_id deterministic", cfg.run_id == load("configs/toy.yaml", args).run_id)
    check("run_id depends on seed", cfg.run_id != replace(cfg, seed=43).run_id)


def check_data():
    for name, f in TOYS.items():
        var = f(200_000, gen(0)).var(0)
        check(f"{name} variance ~ 1", bool(((var - 1).abs() < 0.1).all()), var.tolist())
    xy = (TOYS["checkerboard"](200_000, gen(0)) * 1.15).floor()
    check("checkerboard mass only on dark squares", bool(((xy[:, 0] + xy[:, 1]) % 2 == 0).all()))

    cfg = load("configs/toy.yaml", ["slug=check",
                                    "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"])
    d = training.build_dataset(cfg)
    check("dataset fixed by seed", torch.equal(d.x, training.build_dataset(cfg).x))
    check("dataset differs across seeds", not torch.equal(d.x, training.build_dataset(replace(cfg, seed=43)).x))
    check("reference differs from training set", not torch.equal(training.reference(cfg).x[:100], d.x[:100]))
    clean_only = training.build_dataset(replace(cfg, data={**cfg.data, "keep_noisy": False}))
    check("clean-only keeps exactly the clean samples",
          torch.equal(clean_only.x, d.x[d.sigma == 0]) and bool((clean_only.sigma == 0).all()))

    clean = sample_clean("checkerboard", 200_000, gen(1))
    obs = gaussian(clean, gen(2), sigma=0.3, clean_frac=0.1)
    noisy = obs.sigma > 0
    frac, resid = noisy.float().mean().item(), (obs.x - clean.x)[noisy].std().item()
    check("noisy fraction = 1 - clean_frac", abs(frac - 0.9) < 0.005, f"{frac:.4f}")
    check("noise std = sigma", abs(resid - 0.3) < 0.003, f"{resid:.4f}")
    check("sigma takes values {0, sigma}", torch.equal(obs.sigma.unique(), torch.tensor([0.0, 0.3])))
    check("clean samples untouched", torch.equal(obs.x[~noisy], clean.x[~noisy]))


def check_math():
    x, eps = torch.randn(8, 2), torch.randn(8, 2)
    check("path endpoints", torch.equal(interpolate(x, eps, torch.zeros(8)), x)
          and torch.equal(interpolate(x, eps, torch.ones(8)), eps))

    # With the exact flow and the marginal velocity, the MeanFlow target equals u.
    n, f64 = 4096, torch.float64
    flow = GaussianFlow(a=0.5)
    z = torch.randn(n, 2, dtype=f64)
    a, b = torch.rand(n, dtype=f64), torch.rand(n, dtype=f64)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    r[: n // 4] = t[: n // 4]
    u, u_tgt = target(flow, z, r, t, flow(z, t, t))
    check("MeanFlow target = exact u at the fixed point", torch.allclose(u, u_tgt, atol=1e-8),
          f"max err {(u - u_tgt).abs().max():.1e}")

    # The JVP is the total derivative along the path: compare with finite differences.
    torch.manual_seed(0)
    net = MLP(dim=2, hidden=64, depth=3).double()
    t = 0.2 + 0.6 * torch.rand(n, dtype=f64)
    r = t * torch.rand(n, dtype=f64)
    v = torch.randn(n, 2, dtype=f64)
    h = 1e-7
    fd = (net(z + h * v, r, t + h) - net(z - h * v, r, t - h)) / (2 * h)
    _, u_tgt = target(net, z, r, t, v)
    jvp = (v - u_tgt) / bcast(t - r, z)
    check("JVP = finite-difference du/dt", torch.allclose(jvp, fd, rtol=1e-4, atol=1e-5),
          f"max err {(jvp - fd).abs().max():.1e}")

    # The exact flow maps eps to a * eps in any number of steps; Euler converges to it.
    eps = torch.randn(n, 2, dtype=f64)
    for k in (1, 2, 7):
        check(f"flowmap sampler exact in {k} step(s)",
              torch.allclose(SAMPLERS["flowmap"](flow, eps, k), 0.5 * eps, atol=1e-12))
    err = (SAMPLERS["euler"](flow, eps, 1000) - 0.5 * eps).abs().max().item()
    check("euler sampler converges", err < 1e-2, f"max err {err:.1e}")

    # Flow matching scores the exact velocity below a perturbed one on the same draws.
    batch = Batch(x=0.5 * torch.randn(200_000, 2, generator=gen(3)), sigma=torch.zeros(200_000))
    losses = []
    for scale in (1.0, 1.2):
        torch.manual_seed(0)
        losses.append(flow_matching.loss(lambda *a: scale * flow(*a), batch)[0].item())
    check("flow-matching loss minimal at the true velocity", losses[0] < losses[1], losses)

    p, q = torch.randn(5000, 2, generator=gen(4)), torch.randn(5000, 2, generator=gen(5))
    near, far = energy_distance(p, q), energy_distance(p, q + 1)
    check("energy distance zero on identical sets", abs(energy_distance(p, p)) < 1e-6)
    check("energy distance symmetric", abs(near - energy_distance(q, p)) < 1e-6)
    check("energy distance detects a shift", near < 0.01 and far > 0.3, f"{near:.4f}, {far:.4f}")


def metrics(tmp: Path, name: str) -> list[dict]:
    """The metrics of every record written by run `name`, in order."""
    lines = (tmp / "check.jsonl").read_text().splitlines()
    return [r["metrics"] for r in map(json.loads, lines) if r["config"]["out_root"] == f"{tmp}/{name}"]


def run(tmp: Path, name: str, extra=()):
    args = ["slug=check", f"out_root={tmp}/{name}", "train.steps=2000", "train.eval_every=1000",
            "train.log_every=1000", *extra]
    cfg = load("configs/toy.yaml", args)
    training.train(cfg)
    return cfg


def check_reproducible(tmp: Path):
    run(tmp, "a")
    run(tmp, "b")
    a = metrics(tmp, "a")
    check("identical runs are bitwise reproducible", len(a) == 2 and a == metrics(tmp, "b"), a[-1])

    ref = training.reference(load("configs/toy.yaml", ["slug=check"])).x
    prior = energy_distance(torch.randn(ref.shape, generator=gen(6)), ref)
    final = a[-1]["energy_distance@1"]
    check("MeanFlow learns the checkerboard in 1 step", final < 0.3 * prior,
          f"{final:.4f} vs N(0, I) {prior:.4f}")


def check_resume(tmp: Path):
    """Interrupt right after the first checkpoint is written, then rerun the same config."""
    real_save = torch.save

    def save_then_interrupt(obj, path):
        real_save(obj, path)
        raise KeyboardInterrupt

    torch.save = save_then_interrupt
    try:
        run(tmp, "resume")
    except KeyboardInterrupt:
        pass
    finally:
        torch.save = real_save
    cfg = run(tmp, "resume")
    resumed = metrics(tmp, "resume")[-1]
    check("interrupted run resumes bitwise", resumed == metrics(tmp, "a")[-1], resumed)

    _, net = training.load_run(Path(cfg.out_root) / cfg.slug / cfg.run_id)
    rescored = training.evaluate(net, cfg, training.reference(cfg).to("cuda"))
    check("re-scoring a checkpoint matches its record", rescored == resumed, rescored)


def check_arms(tmp: Path):
    noisy = "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"
    arms = {
        "fm-corrupt-euler": ["objective={name: flow_matching}", "sampler={name: euler, steps: [50]}", noisy],
        "meanflow-clean-only": [noisy, "data.keep_noisy=false"],
        "meanflow-bf16": ["train.precision=bf16"],
    }
    for name, extra in arms.items():
        try:
            run(tmp, name, ["train.steps=300", "train.eval_every=300", *extra])
            m = metrics(tmp, name)[-1]
            check(f"arm runs: {name}", all(math.isfinite(v) for v in m.values()), m)
        except Exception:
            traceback.print_exc()
            check(f"arm runs: {name}", False, "raised")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
          f"{torch.cuda.get_device_name()}", flush=True)
    for fn in (check_config, check_data, check_math):
        section(fn)
    with tempfile.TemporaryDirectory() as tmp:
        training.RUNS = Path(tmp)
        for fn in (check_reproducible, check_resume, check_arms):
            section(fn, Path(tmp))
    print(f"{len(FAILED)} failed: {FAILED}" if FAILED else "all checks passed")
    raise SystemExit(1 if FAILED else 0)
