"""The training loop, shared by every objective and dataset (src/datasets/).

A run lives in `<out_root>/<slug>/<run_id>/` (resolved config, checkpoint and
`true-vs-generated.png` after the last step). At each evaluation milestone the EMA
network is sampled, scored against fresh clean samples (src/metrics.py), checkpointed,
and one record is appended to `<runs_dir>/<slug>.jsonl`. Rerunning the same config
resumes from the checkpoint. Relative paths are taken from the repository root.
"""

import copy
import json
import os
import subprocess
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import yaml

from src.config import REPO, Config, kwargs
from src.corrupt import CORRUPTIONS
from src.data import Batch, batches
from src.datasets import DATASETS
from src.metrics import METRICS
from src.models import MODELS
from src.objectives import OBJECTIVES
from src.sampling import SAMPLERS

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def build_dataset(cfg: Config) -> Batch:
    """The fixed training set: depends only on the data, corruption and seed."""
    gen = torch.Generator().manual_seed(cfg.seed)
    corrupt = partial(CORRUPTIONS[cfg.corruption["name"]], gen=gen, **kwargs(cfg.corruption))
    data = DATASETS[cfg.data["name"]].build(cfg, corrupt, gen)
    return data if cfg.data["keep_noisy"] else data[data.sigma == 0]


def reference(cfg: Config) -> Batch:
    """Fresh clean samples, disjoint in randomness from the training set."""
    gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    return DATASETS[cfg.data["name"]].reference(cfg, cfg.metric["n_samples"], gen)


@torch.no_grad()
def samples(net, cfg: Config, ref: Batch) -> dict:
    """Samples for every sampler step count, from the same seeded noise and without
    touching the training RNG. Class-conditional runs (`objective.guidance`) sample the
    reference's labels. A list of guidance scales `sampler.omega` samples each one, keyed
    "<steps>_w<omega>"; a single scale keys by step count alone."""
    gen = torch.Generator(ref.x.device).manual_seed(cfg.seed)
    eps = torch.randn(ref.x.shape, generator=gen, device=ref.x.device)
    y = ref.cond if cfg.objective["guidance"] else None
    omega = cfg.sampler["omega"]
    if isinstance(omega, list):
        return {f"{steps}_w{w:g}": draw(net, cfg, eps, steps, y, w)
                for steps in cfg.sampler["steps"] for w in omega}
    return {steps: draw(net, cfg, eps, steps, y, omega) for steps in cfg.sampler["steps"]}


def evaluate(net, cfg: Config, ref: Batch) -> dict:
    """Scores the samples of every sampler step count against the reference."""
    return METRICS[cfg.metric["name"]](cfg, samples(net, cfg, ref), ref)


def draw(net, cfg: Config, eps, steps: int, y, omega: float):
    """Samples from `eps` with the configured sampler at guidance scale `omega`, `y` the
    labels or None for the null class, `sampler.batch` noise vectors at a time if set."""
    options = {k: v for k, v in kwargs(cfg.sampler).items() if k not in ("steps", "batch", "omega")}
    size = cfg.sampler.get("batch") or len(eps)
    ys = [None] * len(eps.split(size)) if y is None else y.split(size)
    return torch.cat([SAMPLERS[cfg.sampler["name"]](net, e, steps, c, omega=omega, **options)
                      for e, c in zip(eps.split(size), ys)])


def build_optimizer(net, train: dict):
    """Adam with the official iMF `warmup_const` schedule: linear from 0 over
    `warmup_steps` (the first update has lr 0, as in optax), then constant."""
    opt = torch.optim.Adam(net.parameters(), lr=train["lr"], betas=(0.9, train["adam_b2"]))
    warmup = train["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / warmup) if warmup else 1.0)
    return opt, sched


@torch.no_grad()
def update_ema(ema, net, decay: float):
    """The official update: ema = decay * ema + (1 - decay) * params."""
    for e, p in zip(ema.parameters(), net.parameters()):
        e.lerp_(p, 1 - decay)


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"],
                           cwd=REPO, capture_output=True, text=True)
    return {"git": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def train(cfg: Config):
    device = "cuda"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    run_dir = REPO / cfg.out_root / cfg.slug / cfg.run_id
    runs_file = REPO / cfg.runs_dir / f"{cfg.slug}.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    torch.manual_seed(cfg.seed)

    dataset = DATASETS[cfg.data["name"]]
    data, ref = build_dataset(cfg), reference(cfg).to(device)
    data = data.to(device) if dataset.ON_DEVICE else data
    net = MODELS[cfg.model["name"]](**kwargs(cfg.model)).to(device)
    ema = copy.deepcopy(net).requires_grad_(False)  # starts at the initial weights
    opt, sched = build_optimizer(net, cfg.train)
    loss_fn = partial(OBJECTIVES[cfg.objective["name"]], **kwargs(cfg.objective))
    dtype = DTYPES[cfg.train["precision"]]

    step, elapsed = 0, 0.0
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, elapsed = state["step"], state["elapsed"]
        torch.cuda.set_rng_state(state["rng"].cpu())
        print(f"resumed {run_dir} at step {step}")

    print(f"run {run_dir}: {len(data)} training samples, {int((data.sigma > 0).sum())} noisy")
    stream, start = batches(data, cfg.train["batch_size"]), time.time() - elapsed
    while step < cfg.train["steps"]:
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss, logs = loss_fn(net, next(stream).to(device).resample())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        update_ema(ema, net, cfg.train["ema"])
        step += 1

        if step % cfg.train["log_every"] == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  + " ".join(f"{k} {v.item():.4f}" for k, v in logs.items()), flush=True)
        if step % cfg.train["eval_every"] == 0 or step == cfg.train["steps"]:
            generated = samples(ema, cfg, ref)
            metrics = METRICS[cfg.metric["name"]](cfg, generated, ref)
            elapsed = time.time() - start
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "elapsed": elapsed,
                        "rng": torch.cuda.get_rng_state()}, ckpt)
            record = {"run_id": cfg.run_id, "slug": cfg.slug, **git_state(),
                      "job": os.environ.get("SLURM_JOB_ID"), "seed": cfg.seed, "step": step,
                      "wall_clock_s": round(elapsed, 1), "metrics": metrics, "config": asdict(cfg)}
            with open(runs_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"step {step} " + " ".join(f"{k} {v:.5f}" for k, v in metrics.items()), flush=True)
            if step == cfg.train["steps"]:
                dataset.figure(cfg, data, ref, generated, run_dir / "true-vs-generated.png")


def load_run(run_dir: Path) -> tuple[Config, torch.nn.Module]:
    """A finished or in-progress run's config and EMA network."""
    cfg = Config(**yaml.safe_load((run_dir / "config.yaml").read_text()))
    ema = MODELS[cfg.model["name"]](**kwargs(cfg.model))
    ema.load_state_dict(torch.load(run_dir / "ckpt.pt", map_location="cpu")["ema"])
    return cfg, ema.cuda().eval()
