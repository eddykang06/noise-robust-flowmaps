"""Scores a run's raw (non-EMA) weights, for runs too short for EMA 0.9999 to have moved off its
initial weights. Usage: raw_net_check.py <run_dir> <figure.png>. Run from main's tree."""
import sys
from pathlib import Path

import torch

from src.datasets import DATASETS
from src.metrics import METRICS
from src.training import load_run, reference, samples

run = Path(sys.argv[1])
cfg, net = load_run(run)
net.load_state_dict(torch.load(run / "ckpt.pt", map_location="cuda")["net"])
ref = reference(cfg).to("cuda")
generated = samples(net, cfg, ref)
print("raw net", METRICS[cfg.metric["name"]](cfg, generated, ref))
DATASETS[cfg.data["name"]].figure(cfg, None, ref, generated, Path(sys.argv[2]))
