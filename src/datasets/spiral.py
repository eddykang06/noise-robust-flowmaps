"""The toy: a two-armed spiral, scaled to unit variance per coordinate and observed in
`data.dim` dimensions on a fixed random plane (src/embed.py). Noise is added in the
plane's frame, before the rotation, so per-coordinate noise keeps its shape in the plane."""

import json
import math
from dataclasses import replace

import torch

from src.config import REPO, Config
from src.data import Batch
from src.embed import frame, observe, pad
from src.metrics import wasserstein2
from src.plots import save

ON_DEVICE = True


def spiral(n: int, gen: torch.Generator) -> torch.Tensor:
    """Two-armed spiral with small radial jitter."""
    t = 3 * math.pi * torch.rand(n, generator=gen).sqrt()
    arm = torch.randint(0, 2, (n,), generator=gen) * 2 - 1
    r = t + 0.1 * torch.randn(n, generator=gen)
    xy = arm[:, None] * torch.stack([r * t.cos(), r * t.sin()], 1)
    return xy / 4.71  # per-coordinate std: sqrt((E[t^2] + 0.01) / 2), E[t^2] = 9 pi^2 / 2


def sample_clean(n: int, gen: torch.Generator) -> Batch:
    return Batch(x=spiral(n, gen), sigma=torch.zeros(n))


def build(cfg: Config, corrupt, gen: torch.Generator) -> Batch:
    return observe(corrupt(pad(sample_clean(cfg.data["n"], gen), cfg.data["dim"])))


def reference(cfg: Config, n: int, gen: torch.Generator) -> Batch:
    return observe(pad(sample_clean(n, gen), cfg.data["dim"]))


def figure_run(cfg: Config, data: Batch, ref: Batch, generated: dict) -> dict:
    """What `src.plots` draws for one run, in the plane: its records so far and the
    metric's floor, the score of a second, independent clean sample."""
    runs_file = REPO / cfg.runs_dir / f"{cfg.slug}.jsonl"
    records = [r for r in map(json.loads, runs_file.read_text().splitlines())
               if r["run_id"] == cfg.run_id]
    other = sample_clean(len(ref), torch.Generator().manual_seed(20_000 + cfg.seed))
    plane = lambda x: frame(x)[:, :2]
    floor = wasserstein2(other.x, plane(ref.x))
    return {"label": cfg.run_id, "data": replace(data, x=plane(data.x)), "true": plane(ref.x),
            "generated": {steps: plane(x) for steps, x in generated.items()},
            "records": records, "metric": cfg.metric["name"], "floor": floor}


def figure(cfg: Config, data: Batch, ref: Batch, generated: dict, path):
    save([figure_run(cfg, data, ref, generated)], path)
