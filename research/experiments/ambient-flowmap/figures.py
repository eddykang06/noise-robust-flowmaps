"""This direction's figures, as matplotlib PNGs, from the re-scoring jobs'
logs (more-steps.sbatch: eval.txt and plane-samples.json).

    python research/experiments/ambient-flowmap/figures.py <out_dir>

Each set gets two figures: W2 against sampling steps, and a grid of samples. Each entry of SETS
is one sweep: its runs (label -> (run name, key in plane-samples.json, the
<key> of more-steps' RUNS pairs)), the re-scoring jobs holding them, and how its series are coloured (ordinal: a light-to-dark blue ramp for D; categorical:
fixed hue order for arms). A set is drawn only once all its inputs exist.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path("/n/scratch/users/e/edk005/nrfm/ambient-flowmap/logs")
STEPS = [1, 2, 3, 4, 8]
W2_LABEL = "W2 to the true spiral (lower is better)"
W2_NOTE = ("W2: Wasserstein-2 distance between 4096 generated and 4096 true samples, "
           "measured in the spiral's 2-D plane.")
INK, MUTED, RULE = "#26282c", "#6b6f76", "#e3e4e6"
BLUES = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]

SETS = {
    "dim-sweep-100k": {
        "title": "Clean exact iMF, spiral observed in D dimensions, 100k steps",
        "runs": {"D = 2": ("imf-s42-7e4d5767", "2"), "D = 8": ("imf-s42-81dff414", "8"), "D = 16": ("imf-s42-4ec28c3d", "16"),
                 "D = 32": ("imf-s42-95e1a81a", "32"), "D = 64": ("imf-s42-f5958283", "64")},
        "jobs": ["60033355", "60038059"], "colors": BLUES},
    "arms-d64": {
        "title": "Five arms at D = 64, 100k steps",
        "runs": {"1 clean": ("imf-s42-f5958283", "arm1"), "2 naive, Gaussian": ("imf-s42-cda379ef", "arm2"),
                 "3 ambient, Gaussian": ("ambient_imf-s42-f3ed2000", "arm3"), "4 ambient, uniform": ("ambient_imf-s42-c8f8ec04", "arm4"),
                 "5 naive, uniform": ("imf-s42-2500a6d4", "arm5")},
        "jobs": ["60108782"], "colors": HUES},
}
# Fixed-sigma sweep, 75% clean (60187230/34/87/88, re-scored 60194400), one set per sigma;
# arm 1 is the clean run re-scored in 60108782. Keys are <G|U><sigma*100><n|a>.
SWEEP = {"0.05": ["imf-s42-d2f56231", "ambient_imf-s42-7da581e8", "ambient_imf-s42-5d62af51", "imf-s42-f1bcb92c"],
         "0.1": ["imf-s42-44ec811d", "ambient_imf-s42-d9acc2f3", "ambient_imf-s42-72c3ce86", "imf-s42-3542192e"],
         "0.2": ["imf-s42-a42acddd", "ambient_imf-s42-a41185d9", "ambient_imf-s42-93ed43a6", "imf-s42-3d397500"],
         "0.33": ["imf-s42-8fc1b105", "ambient_imf-s42-a013d25f", "ambient_imf-s42-cc685b61", "imf-s42-d3226d19"]}
ARMS = ["1 clean", "2 naive, Gaussian", "3 ambient, Gaussian", "4 ambient, uniform", "5 naive, uniform"]
for sigma, runs in SWEEP.items():
    tag = f"{round(float(sigma) * 100):02d}"
    keys = [f"g{tag}n", f"g{tag}a", f"u{tag}a", f"u{tag}n"]
    SETS[f"sigma-{sigma}"] = {
        "title": f"Five arms, 75% clean, sigma = {sigma}",
        "runs": dict(zip(ARMS, [("imf-s42-f5958283", "arm1"), *zip(runs, keys)])),
        "jobs": ["60108782", "60194400"], "colors": HUES}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 5,
})


def load(spec):
    """Re-scored metrics and plane samples for each run of a set."""
    ids = {run: label for label, (run, _) in spec["runs"].items()}
    evals, samples = {}, {}
    for job in spec["jobs"]:
        for line in (LOGS / job / "eval.txt").read_text().splitlines():
            run, metrics = line.split(" ", 1)
            if run in ids:
                evals[ids[run]] = json.loads(metrics)
        by_dim = json.loads((LOGS / job / "plane-samples.json").read_text())
        samples.update({label: by_dim[key] for label, (_, key) in spec["runs"].items() if key in by_dim})
    return evals, samples


def fig_steps(name, spec, evals, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (label, ev), color in zip(evals.items(), spec["colors"]):
        ax.plot(STEPS, [ev[f"wasserstein2@{s}"] for s in STEPS], color=color, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel("sampling steps used to generate")
    ax.set_ylabel(W2_LABEL)
    ax.set_title("Sample quality vs. number of sampling steps (final checkpoint)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.text(0.01, -0.03, W2_NOTE, fontsize=8, color=MUTED)
    fig.savefig(out / f"{name}-steps.png")
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
            ax.scatter([p[0] for p in x], [p[1] for p in x], s=1.2, linewidths=0, alpha=0.75,
                       color=MUTED if k == "true" else color, rasterized=True)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[], aspect="equal")
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
            if k != "true":
                ax.text(0.04, 0.04, f"W2 = {evals[label][f'wasserstein2@{k}']:.3f}", transform=ax.transAxes,
                        fontsize=8, color=INK)
            if i == 0:
                ax.set_title("true spiral" if k == "true" else f"{k} sampling step{'s' if k != '1' else ''}",
                             fontsize=9, loc="center", fontweight="normal", color=MUTED)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
    fig.suptitle("Generated samples, projected onto the spiral's plane", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.01, W2_NOTE, fontsize=8, color=MUTED)
    fig.savefig(out / f"{name}-samples.png")
    plt.close(fig)


def fig_sigma(out):
    """W2 at 1 and 2 steps against sigma, one line per arm, across the fixed-sigma sweep."""
    sets = [SETS[f"sigma-{sigma}"] for sigma in SWEEP]
    evals = [load(spec)[0] for spec in sets]
    sigmas = [float(s) for s in SWEEP]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, k in zip(axes, [1, 2]):
        for label, color in zip(ARMS, HUES):
            ax.plot(sigmas, [ev[label][f"wasserstein2@{k}"] for ev in evals], color=color, marker="o", label=label)
        ax.set_xticks(sigmas, list(SWEEP))
        ax.set_xlabel("noise level sigma of the corrupted 25%")
        ax.set_title(f"{k} sampling step{'s' if k > 1 else ''}")
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel(W2_LABEL)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.text(0.01, -0.04, W2_NOTE + " Seed 42 only.", fontsize=8, color=MUTED)
    fig.savefig(out / "sweep-sigma.png")
    plt.close(fig)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in SETS.items():
        if not spec["runs"] or not all((LOGS / j / "eval.txt").exists() for j in spec["jobs"]):
            print(f"{name}: inputs missing, skipped")
            continue
        evals, samples = load(spec)
        evals = {label: evals[label] for label in spec["runs"]}
        fig_steps(name, spec, evals, out)
        fig_samples(name, spec, evals, samples, out)
        print(f"{name}: wrote {name}-steps.png, -samples.png")
    if (LOGS / "60194400" / "eval.txt").exists():
        fig_sigma(out)
        print("wrote sweep-sigma.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
