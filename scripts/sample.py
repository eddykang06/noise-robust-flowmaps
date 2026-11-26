"""Draw samples from a run's EMA network into <run_dir>/samples-<steps>step.pt, a dict of the
samples `x` and their labels `y`: balanced over the classes (as iMF) for class-conditional runs,
None otherwise.

    python -m scripts.sample <run_dir> --n 10000 --steps 1 --omega 1.0 --seed 0
"""

import argparse
from pathlib import Path

import torch

from src.training import draw, load_run, reference

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--omega", type=float, required=True, help="guidance scale")
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

cfg, net = load_run(args.run_dir)
shape = reference(cfg).x.shape[1:]
gen = torch.Generator("cuda").manual_seed(args.seed)
eps = torch.randn((args.n, *shape), generator=gen, device="cuda")
y = torch.arange(args.n, device="cuda") % net.num_classes if cfg.objective["guidance"] else None
x = draw(net, cfg, eps, args.steps, y, args.omega)
torch.save({"x": x.cpu(), "y": None if y is None else y.cpu()}, args.run_dir / f"samples-{args.steps}step.pt")
