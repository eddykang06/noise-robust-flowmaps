"""Train one run.

    python -m scripts.train --config configs/toy.yaml slug=<slug> seed=42 [key=value ...]
"""

import argparse

from src.config import load
from src.training import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("overrides", nargs="*", help="dotted key=value, e.g. objective.ratio=0.5")
args = parser.parse_args()
train(load(args.config, args.overrides))
