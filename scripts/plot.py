"""One row per run: its training data (noisy samples in orange), the clean
reference, the EMA network's samples at each sampler step count, and the metric
over training against the floor of two independent clean samples.

    python -m scripts.plot <run_dir> [<run_dir> ...] --out <figure.png>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import sample_clean
from src.metrics import METRICS
from src.training import REPO, build_dataset, load_run, reference, samples

LIM = 3.0  # toys have unit variance per coordinate


def scatter(ax, x, title, color="tab:blue"):
    x = x.cpu()
    ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.4, color=color, linewidths=0)
    ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[], title=title)


def plot(run_dirs: list[Path], out: Path):
    cfgs_nets = [load_run(d) for d in run_dirs]
    ncols = 3 + len(cfgs_nets[0][0].sampler["steps"])
    fig, axes = plt.subplots(len(run_dirs), ncols, figsize=(3 * ncols, 3 * len(run_dirs)), squeeze=False)
    for row, (cfg, net) in zip(axes, cfgs_nets):
        name = cfg.metric["name"]
        ref = reference(cfg).to("cuda")
        gen = torch.Generator().manual_seed(20_000 + cfg.seed)
        floor = METRICS[name](sample_clean(cfg.data["name"], cfg.metric["n_samples"], gen).x, ref.x)
        runs_file = REPO / cfg.runs_dir / f"{cfg.slug}.jsonl"
        records = [r for r in map(json.loads, runs_file.read_text().splitlines()) if r["run_id"] == cfg.run_id]
        last = records[-1]["metrics"]

        data = build_dataset(cfg)
        noisy = data.sigma > 0
        title = f"training data ({int(noisy.sum())} noisy)"
        scatter(row[0], data.x[noisy], title, color="tab:orange")
        scatter(row[0], data.x[~noisy], title)
        scatter(row[1], ref.x, "clean reference")
        for ax, (steps, x) in zip(row[2:], samples(net, cfg, ref).items()):
            scatter(ax, x, f"{steps}-step, {name} {last[f'{name}@{steps}']:.3f}")
        ax = row[-1]
        for key in last:
            ax.plot([r["step"] for r in records], [r["metrics"][key] for r in records], marker="o", label=key)
        ax.axhline(floor, color="gray", linestyle="--", label=f"floor {floor:.3f}")
        ax.set(yscale="log", xlabel="training step", title=name)
        ax.legend(fontsize=7)
        row[0].set_ylabel(cfg.run_id, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run_dirs, args.out)
