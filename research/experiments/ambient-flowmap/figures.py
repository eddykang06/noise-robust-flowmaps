"""This direction's figures, as matplotlib PNGs, from the runs file and the re-scoring jobs'
logs (more-steps.sbatch: eval.txt and plane-samples.json).

    python research/experiments/ambient-flowmap/figures.py <out_dir>

Each entry of SETS is one sweep: its runs (label -> (run hash, key in plane-samples.json, the
<key> of more-steps' RUNS pairs)), the re-scoring jobs holding them, and how its series are coloured (ordinal: a light-to-dark blue ramp for D; categorical:
fixed hue order for arms). A set is drawn only once all its inputs exist.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
LOGS = Path("/n/scratch/users/e/edk005/nrfm/ambient-flowmap/logs")
STEPS = [1, 2, 3, 4, 8]
PASS, FLOOR = 0.15, 0.122
INK, MUTED, RULE = "#26282c", "#6b6f76", "#e3e4e6"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]

SETS = {
    "dim-sweep-100k": {
        "title": "Clean exact iMF, spiral observed in D dimensions, 100k steps",
        "runs": {"D = 2": ("7e4d5767", "2"), "D = 8": ("81dff414", "8"), "D = 16": ("4ec28c3d", "16"),
                 "D = 32": ("95e1a81a", "32"), "D = 64": ("f5958283", "64")},
        "jobs": ["60033355", "60038059"], "colors": BLUES},
    "dim-sweep-50k": {
        "title": "Clean exact iMF, spiral observed in D dimensions, 50k steps",
        "runs": {"D = 2": ("61bb6b98", "2"), "D = 8": ("d84ff5b3", "8"), "D = 16": ("16b83d21", "16"),
                 "D = 512": ("c1d083a4", "512")},
        "jobs": ["60026233"], "colors": BLUES[:3] + BLUES[-1:]},
    "arms-d64": {
        "title": "Five arms at D = 64, 100k steps",
        "runs": {},  # filled from job 60098738's arm logs once it finishes
        "jobs": [], "colors": HUES},
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 5,
})


def load(spec):
    """Training records, re-scored metrics and plane samples for each run of a set."""
    ids = {f"imf-s42-{h}": label for label, (h, _) in spec["runs"].items()}
    curves = {label: [] for label in spec["runs"]}
    for line in (REPO / "research/runs/ambient-flowmap.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["run_id"] in ids:
            curves[ids[r["run_id"]]].append((r["step"], r["metrics"]))
    evals, samples = {}, {}
    for job in spec["jobs"]:
        for line in (LOGS / job / "eval.txt").read_text().splitlines():
            run, metrics = line.split(" ", 1)
            evals[ids[run]] = json.loads(metrics)
        by_dim = json.loads((LOGS / job / "plane-samples.json").read_text())
        samples.update({label: by_dim[key] for label, (_, key) in spec["runs"].items() if key in by_dim})
    return {k: sorted(v, key=lambda r: r[0]) for k, v in curves.items()}, evals, samples


def reference_lines(ax):
    ax.axhline(PASS, color="#1baf7a", linestyle="--", linewidth=1.2)
    ax.axhline(FLOOR, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(1.0, PASS, " pass 0.15", transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=8, color=MUTED)
    ax.text(1.0, FLOOR, " floor 0.122", transform=ax.get_yaxis_transform(), va="top", ha="right",
            fontsize=8, color=MUTED)


def end_labels(ax, x, ys, texts, gap):
    """Direct labels right of x, at ys nudged apart by at least `gap` (data units)."""
    placed = {}
    for i in sorted(range(len(ys)), key=lambda i: ys[i]):
        y = ys[i] if not placed else max(ys[i], max(placed.values()) + gap)
        placed[i] = y
        ax.annotate(texts[i], (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False)


def fig_steps(name, spec, evals, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (label, ev), color in zip(evals.items(), spec["colors"]):
        ax.plot(STEPS, [ev[f"wasserstein2@{s}"] for s in STEPS], color=color, marker="o", label=label)
    finals = [ev["wasserstein2@8"] for ev in evals.values()]
    end_labels(ax, STEPS[-1], finals, [f"{label}  {w:.3f}" for label, w in zip(evals, finals)],
               gap=0.03 * max(ev["wasserstein2@1"] for ev in evals.values()))
    reference_lines(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel("sampling steps")
    ax.set_ylabel("W2 in the plane (lower is better)")
    ax.set_title("W2 by sampling steps, final checkpoint")
    ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(0.8, 1))
    fig.savefig(out / f"{name}-steps.png")
    plt.close(fig)


def fig_training(name, spec, curves, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, s in zip(axes, [1, 2]):
        for (label, recs), color in zip(curves.items(), spec["colors"]):
            ax.plot([r[0] / 1000 for r in recs], [r[1][f"wasserstein2@{s}"] for r in recs],
                    color=color, marker="o", label=label)
        reference_lines(ax)
        ax.set_title(f"{s} sampling step{'s' if s > 1 else ''}")
        ax.set_xlabel("training step (thousands)")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("W2 in the plane")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.savefig(out / f"{name}-training.png")
    plt.close(fig)


def fig_samples(name, spec, evals, samples, out, lim=2.3):
    cols = ["true", *map(str, STEPS)]
    rows = list(samples)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.55 * len(cols), 1.55 * len(rows)),
                             squeeze=False)
    for i, (label, color) in enumerate(zip(rows, spec["colors"])):
        for j, k in enumerate(cols):
            ax = axes[i, j]
            x = samples[label][k][:2048]
            ax.scatter([p[0] for p in x], [p[1] for p in x], s=0.6, linewidths=0, alpha=0.6,
                       color=MUTED if k == "true" else color, rasterized=True)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[], aspect="equal")
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
            if k != "true":
                ax.text(0.04, 0.04, f"{evals[label][f'wasserstein2@{k}']:.3f}", transform=ax.transAxes,
                        fontsize=8, color=INK)
            if i == 0:
                ax.set_title("true" if k == "true" else f"{k} step{'s' if k != '1' else ''}",
                             fontsize=9, loc="center", fontweight="normal", color=MUTED)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
    fig.suptitle("Samples in the spiral's plane (W2 bottom left)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / f"{name}-samples.png")
    plt.close(fig)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in SETS.items():
        if not spec["runs"] or not all((LOGS / j / "eval.txt").exists() for j in spec["jobs"]):
            print(f"{name}: inputs missing, skipped")
            continue
        curves, evals, samples = load(spec)
        evals = {label: evals[label] for label in spec["runs"]}
        fig_steps(name, spec, evals, out)
        fig_training(name, spec, curves, out)
        fig_samples(name, spec, evals, samples, out)
        print(f"{name}: wrote {name}-steps.png, -training.png, -samples.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
