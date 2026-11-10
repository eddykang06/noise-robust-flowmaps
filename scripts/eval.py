"""Re-score a run's latest checkpoint, optionally with other step counts.

    python -m scripts.eval <run_dir> [--steps 1 2 4]
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.training import evaluate, load_run, reference

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--steps", type=int, nargs="+")
args = parser.parse_args()

cfg, net = load_run(args.run_dir)
if args.steps:
    cfg = replace(cfg, sampler={**cfg.sampler, "steps": args.steps})
print(json.dumps(evaluate(net, cfg, reference(cfg).to("cuda"))))
