"""The training loop, shared by every objective.

A run lives in `<out_root>/<slug>/<run_id>/` (resolved config and checkpoint).
At each evaluation milestone the EMA network is sampled, scored against fresh
clean samples, checkpointed, and one record is appended to
`research/runs/<slug>.jsonl`. Rerunning the same config resumes from the
checkpoint.
"""

import json
import os
import subprocess
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.config import Config, kwargs
from src.corrupt import CORRUPTIONS
from src.data import Batch, batches, sample_clean
from src.metrics import METRICS
from src.models import MODELS
from src.objectives import OBJECTIVES
from src.sampling import SAMPLERS

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "research" / "runs"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def build_dataset(cfg: Config) -> Batch:
    """The fixed training set: depends only on the data, corruption and seed."""
    gen = torch.Generator().manual_seed(cfg.seed)
    data = sample_clean(cfg.data["name"], cfg.data["n"], gen)
    data = CORRUPTIONS[cfg.corruption["name"]](data, gen, **kwargs(cfg.corruption))
    return data if cfg.data["keep_noisy"] else data[data.sigma == 0]


def reference(cfg: Config) -> Batch:
    """Fresh clean samples, disjoint in randomness from the training set."""
    gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    return sample_clean(cfg.data["name"], cfg.metric["n_samples"], gen)


@torch.no_grad()
def evaluate(net, cfg: Config, ref: Batch) -> dict:
    """Scores every sampler step count on the same seeded noise, without
    touching the training RNG."""
    sampler, metric = SAMPLERS[cfg.sampler["name"]], METRICS[cfg.metric["name"]]
    gen = torch.Generator(ref.x.device).manual_seed(cfg.seed)
    eps = torch.randn(ref.x.shape, generator=gen, device=ref.x.device)
    return {f"{cfg.metric['name']}@{steps}": metric(sampler(net, eps, steps, ref.cond), ref.x)
            for steps in cfg.sampler["steps"]}


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"],
                           cwd=REPO, capture_output=True, text=True)
    return {"git": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def train(cfg: Config):
    device = "cuda"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    run_dir = Path(cfg.out_root) / cfg.slug / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    torch.manual_seed(cfg.seed)

    data, ref = build_dataset(cfg).to(device), reference(cfg).to(device)
    net = MODELS[cfg.model["name"]](**kwargs(cfg.model)).to(device)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(cfg.train["ema"]))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train["lr"])
    loss_fn = partial(OBJECTIVES[cfg.objective["name"]], **kwargs(cfg.objective))
    dtype = DTYPES[cfg.train["precision"]]

    step, elapsed = 0, 0.0
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step, elapsed = state["step"], state["elapsed"]
        torch.cuda.set_rng_state(state["rng"].cpu())
        print(f"resumed {run_dir} at step {step}")

    print(f"run {run_dir}: {len(data)} training samples, {int((data.sigma > 0).sum())} noisy")
    stream, start = batches(data, cfg.train["batch_size"]), time.time() - elapsed
    while step < cfg.train["steps"]:
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss, logs = loss_fn(net, next(stream))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update_parameters(net)
        step += 1

        if step % cfg.train["log_every"] == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  + " ".join(f"{k} {v.item():.4f}" for k, v in logs.items()), flush=True)
        if step % cfg.train["eval_every"] == 0 or step == cfg.train["steps"]:
            metrics = evaluate(ema.module, cfg, ref)
            elapsed = time.time() - start
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "elapsed": elapsed, "rng": torch.cuda.get_rng_state()}, ckpt)
            record = {"run_id": cfg.run_id, "slug": cfg.slug, **git_state(),
                      "job": os.environ.get("SLURM_JOB_ID"), "seed": cfg.seed, "step": step,
                      "wall_clock_s": round(elapsed, 1), "metrics": metrics, "config": asdict(cfg)}
            with open(RUNS / f"{cfg.slug}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"step {step} " + " ".join(f"{k} {v:.5f}" for k, v in metrics.items()), flush=True)


def load_run(run_dir: Path) -> tuple[Config, torch.nn.Module]:
    """A finished or in-progress run's config and EMA network."""
    cfg = Config(**yaml.safe_load((run_dir / "config.yaml").read_text()))
    net = MODELS[cfg.model["name"]](**kwargs(cfg.model))
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(cfg.train["ema"]))
    ema.load_state_dict(torch.load(run_dir / "ckpt.pt", map_location="cpu")["ema"])
    return cfg, ema.module.cuda().eval()
