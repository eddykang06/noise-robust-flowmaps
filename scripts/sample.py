"""Draw samples from a run's EMA network into <run_dir>/samples-<steps>step.pt.

    python -m scripts.sample <run_dir> --n 10000 --steps 1 --seed 0
"""

import argparse
from pathlib import Path

import torch

from src.sampling import SAMPLERS
from src.training import load_run, reference

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

cfg, net = load_run(args.run_dir)
shape = reference(cfg).x.shape[1:]
gen = torch.Generator("cuda").manual_seed(args.seed)
eps = torch.randn((args.n, *shape), generator=gen, device="cuda")
x = SAMPLERS[cfg.sampler["name"]](net, eps, args.steps)
torch.save(x.cpu(), args.run_dir / f"samples-{args.steps}step.pt")
