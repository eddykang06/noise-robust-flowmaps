"""True vs generated figure for one or more finished toy runs, one row each
(see src/plots.py); every toy run also writes its own at the end of training.

    python -m scripts.plot <run_dir> [<run_dir> ...] --out <figure.png>
"""

import argparse
from pathlib import Path

from src.plots import save
from src.training import build_dataset, figure_run, load_run, reference, samples

parser = argparse.ArgumentParser()
parser.add_argument("run_dirs", type=Path, nargs="+")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

runs = []
for run_dir in args.run_dirs:
    cfg, net = load_run(run_dir)
    ref = reference(cfg).to("cuda")
    runs.append(figure_run(cfg, build_dataset(cfg), ref, samples(net, cfg, ref)))
save(runs, args.out)
