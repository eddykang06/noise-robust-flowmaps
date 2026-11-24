"""End-to-end behaviour of a source tree on this direction's runs, for checking that the
consolidated main (configs/imf.yaml, configs/ambient_imf.yaml) behaves bitwise like this
branch (spiral-imf.yaml, spiral-ambient.yaml). Run from the tree's root:

    python <this file> old|new <out.json> <tmp_dir>

Per case: the run id, digests of the training set and the reference, and a 200-step
training run (two evaluations, checkpoint, figure) with its metrics and weight digests.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

from src.config import load
from src.training import build_dataset, reference, train

MODE, OUT, TMP = sys.argv[1], sys.argv[2], Path(sys.argv[3])
CONFIGS = {"old": {"imf": "research/experiments/ambient-flowmap/spiral-imf.yaml",
                   "ambient": "research/experiments/ambient-flowmap/spiral-ambient.yaml"},
           "new": {"imf": "configs/imf.yaml", "ambient": "configs/ambient_imf.yaml"}}[MODE]
BASE = ["slug=ambient-flowmap", "data.dim=64", "model.dim=64", "data.n=100000"]
G, U = ("corruption={name: %s, sigma: 1.0, clean_frac: 0.75}" % n for n in ("gaussian", "uniform"))
CASES = {  # name: (objective, overrides); the first five are the sigma = 1.0, n = 100k runs
    "clean": ("imf", []),
    "imf-gaussian": ("imf", [G]),
    "ambient-gaussian": ("ambient", [G]),
    "ambient-uniform": ("ambient", [U]),
    "imf-uniform": ("imf", [U]),
    "imf-per-sample-var": ("imf", ["data.n=10000", "corruption={name: gaussian_uniform_var, var_max: 1.0, clean_frac: 0.33}"]),
    "ambient-box-var": ("ambient", ["data.n=10000", "corruption={name: box_uniform_var, var_max: 1.0, clean_frac: 0.33}"]),
    "clean-only": ("imf", [G, "data.keep_noisy=false"]),
    "2-D": ("imf", ["data.dim=2", "model.dim=2", "data.n=10000"]),
}


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


out = {}
for name, (objective, extra) in CASES.items():
    cfg = load(CONFIGS[objective], [*BASE, *extra])
    data, ref = build_dataset(cfg), reference(cfg)
    case = {"run_id": cfg.run_id, "run_id_eval2k": load(CONFIGS[objective], [*BASE, *extra, "train.eval_every=2000"]).run_id,
            "data": digest(data.x, data.sigma), "reference": digest(ref.x)}
    short = load(CONFIGS[objective], [*BASE, *extra, "train.steps=200", "train.eval_every=100",
                                      f"out_root={TMP}/{MODE}/{name}", f"runs_dir={TMP}/{MODE}/{name}"])
    train(short)
    run_dir = Path(short.out_root) / short.slug / short.run_id
    state = torch.load(run_dir / "ckpt.pt", map_location="cpu")
    records = [json.loads(l) for l in (Path(short.runs_dir) / f"{short.slug}.jsonl").read_text().splitlines()]
    case.update(metrics=[r["metrics"] for r in records], net=digest(*state["net"].values()),
                ema=digest(*state["ema"].values()), figure=(run_dir / "true-vs-generated.png").stat().st_size > 0)
    out[name] = case
    print(name, json.dumps(case), flush=True)
json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
