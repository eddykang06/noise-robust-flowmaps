"""This direction's figures, as matplotlib PNGs, from the re-scoring jobs'
logs (more-steps.sbatch: eval.txt and plane-samples.json).

    python research/experiments/ambient-flowmap/figures.py <out_dir>

Each set gets two figures: W2 against sampling steps, and a grid of samples. Each entry of SETS
is one sweep: its runs (label -> (run name, key in plane-samples.json, the <key> of
more-steps' RUNS pairs)), the re-scoring jobs holding them, the metric key it reports, and
how its series are coloured (ordinal: a blue ramp for D; categorical: fixed hue order for
arms). A set is drawn only once all its inputs exist. Since 2026-11-20 (human) the metric is
W2 in all D dimensions (`wasserstein2_full`); the two older sets predate it and stay in-plane.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path("/n/scratch/users/e/edk005/nrfm/ambient-flowmap/logs")
STEPS = [1, 2, 3, 4, 8]
FULL = "wasserstein2_full"
W2_LABEL = {"wasserstein2": "W2 in the plane (lower is better)", FULL: "W2 in 64-D (lower is better)"}
W2_NOTE = {"wasserstein2": "W2: Wasserstein-2 distance between 4096 generated and 4096 true samples, "
                           "measured in the spiral's 2-D plane.",
           FULL: "W2: Wasserstein-2 distance between 4096 generated and 4096 true samples in all 64 "
                 "dimensions. Samples are drawn projected onto the spiral's plane."}
INK, MUTED, RULE = "#26282c", "#6b6f76", "#e3e4e6"
BLUES = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]

SETS = {
    "dim-sweep-100k": {
        "title": "Clean exact iMF, spiral observed in D dimensions, 100k steps",
        "runs": {"D = 2": ("imf-s42-7e4d5767", "2"), "D = 8": ("imf-s42-81dff414", "8"), "D = 16": ("imf-s42-4ec28c3d", "16"),
                 "D = 32": ("imf-s42-95e1a81a", "32"), "D = 64": ("imf-s42-f5958283", "64")},
        "jobs": ["60033355", "60038059"], "metric": "wasserstein2", "colors": BLUES},
    "arms-d64": {
        "title": "Five arms at D = 64, 100k steps",
        "runs": {"1 clean": ("imf-s42-f5958283", "arm1"), "2 naive, Gaussian": ("imf-s42-cda379ef", "arm2"),
                 "3 ambient, Gaussian": ("ambient_imf-s42-f3ed2000", "arm3"), "4 ambient, uniform": ("ambient_imf-s42-c8f8ec04", "arm4"),
                 "5 naive, uniform": ("imf-s42-2500a6d4", "arm5")},
        "jobs": ["60108782"], "metric": "wasserstein2", "colors": HUES},
}
# Fixed-sigma sweeps, 75% clean, one set per sigma, all five arms re-scored in 64-D (arm 1 is the
# clean run 60038059). Keys are <g|u><sigma*100><n|a>. sigma -> (re-scoring job, runs in key order):
# 0.05-0.33 trained by 60187230/34/87/88 and re-scored by 60204483; 0.25-1.0 trained by
# 60207440-43 and re-scored by 60224909.
SWEEP = {"0.05": ("60204483", ["imf-s42-d2f56231", "ambient_imf-s42-7da581e8", "ambient_imf-s42-5d62af51", "imf-s42-f1bcb92c"]),
         "0.1": ("60204483", ["imf-s42-44ec811d", "ambient_imf-s42-d9acc2f3", "ambient_imf-s42-72c3ce86", "imf-s42-3542192e"]),
         "0.2": ("60204483", ["imf-s42-a42acddd", "ambient_imf-s42-a41185d9", "ambient_imf-s42-93ed43a6", "imf-s42-3d397500"]),
         "0.25": ("60224909", ["imf-s42-082714c4", "ambient_imf-s42-ebdd3392", "ambient_imf-s42-535d6da3", "imf-s42-fed5112d"]),
         "0.33": ("60204483", ["imf-s42-8fc1b105", "ambient_imf-s42-a013d25f", "ambient_imf-s42-cc685b61", "imf-s42-d3226d19"]),
         "0.5": ("60224909", ["imf-s42-74d62519", "ambient_imf-s42-59feb107", "ambient_imf-s42-73b06472", "imf-s42-2692b6ff"]),
         "0.75": ("60224909", ["imf-s42-8ef229cc", "ambient_imf-s42-146c8b8b", "ambient_imf-s42-3aa7de36", "imf-s42-947b3690"]),
         "1.0": ("60224909", ["imf-s42-bf728638", "ambient_imf-s42-ca865094", "ambient_imf-s42-5412a8d1", "imf-s42-e56b68a9"])}
CLEAN_JOB = "60204483"  # holds arm 1 (clean iMF) for every sigma
ARMS = ["1 clean", "2 naive, Gaussian", "3 ambient, Gaussian", "4 ambient, uniform", "5 naive, uniform"]
for sigma, (job, runs) in SWEEP.items():
    tag = f"{round(float(sigma) * 100):02d}"
    keys = [f"g{tag}n", f"g{tag}a", f"u{tag}a", f"u{tag}n"]
    SETS[f"sigma-{sigma}"] = {
        "title": f"Five arms, 75% clean, sigma = {sigma}",
        "runs": dict(zip(ARMS, [("imf-s42-f5958283", "arm1"), *zip(runs, keys)])),
        "jobs": sorted({CLEAN_JOB, job}), "metric": FULL, "colors": HUES}

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
        ax.plot(STEPS, [ev[f"{spec['metric']}@{s}"] for s in STEPS], color=color, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel("sampling steps used to generate")
    ax.set_ylabel(W2_LABEL[spec["metric"]])
    ax.set_title("Sample quality vs. number of sampling steps (final checkpoint)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.text(0.01, -0.03, W2_NOTE[spec["metric"]], fontsize=8, color=MUTED)
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
                ax.text(0.04, 0.04, f"W2 = {evals[label][spec['metric'] + '@' + k]:.3f}", transform=ax.transAxes,
                        fontsize=8, color=INK)
            if i == 0:
                ax.set_title("true spiral" if k == "true" else f"{k} sampling step{'s' if k != '1' else ''}",
                             fontsize=9, loc="center", fontweight="normal", color=MUTED)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
    fig.suptitle("Generated samples, projected onto the spiral's plane", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.01, W2_NOTE[spec["metric"]], fontsize=8, color=MUTED)
    fig.savefig(out / f"{name}-samples.png")
    plt.close(fig)


def sweep():
    """Metrics and samples of every arm at every sigma of the fixed-sigma sweeps."""
    return {sigma: load(SETS[f"sigma-{sigma}"]) for sigma in SWEEP}


def title(k):
    return f"{k} sampling step{'s' if k > 1 else ''}"


def fig_sigma(out, sets, arms, name, steps=(1, 2, 8), metric=FULL):
    """W2 (`metric`) against sigma at each step count in `steps`, one line per arm, clean iMF dashed."""
    sigmas = [float(s) for s in SWEEP]
    clean = sets[next(iter(SWEEP))][0][ARMS[0]]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.3 * len(steps), 3.2), sharey=True)
    for ax, k in zip(axes, steps):
        ax.axhline(clean[f"{metric}@{k}"], color=MUTED, linestyle="--", linewidth=1.2, label="iMF, clean data")
        for label, short in arms.items():
            ax.plot(sigmas, [sets[s][0][label][f"{metric}@{k}"] for s in SWEEP], color=HUES[ARMS.index(label)],
                    marker="o", label=short)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("noise level σ")
        ax.set_title(title(k), fontweight="normal")
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel(W2_LABEL[metric])
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.savefig(out / f"{name}-{'w2' if metric == FULL else 'plane'}.png")
    plt.close(fig)


def fig_breakdown(out, sets, arms, name, steps=(1, 8)):
    """W2² in 64-D split into its in-plane part (W2² in the plane) and its off-plane part (mean
    squared distance from the plane), exact because the clean data lie on the plane; one bar
    per arm at each sigma, clean iMF first."""
    sigmas = list(SWEEP)
    width = 0.8 / len(arms)
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 3.4), sharey=False)
    for ax, k in zip(axes, steps):
        clean = sets[sigmas[0]][0][ARMS[0]]
        bars = [("clean", -1, clean, MUTED)]
        for i, label in enumerate(arms):
            for j, sigma in enumerate(sigmas):
                bars.append((label, j + (i - (len(arms) - 1) / 2) * width, sets[sigma][0][label],
                             HUES[ARMS.index(label)]))
        for label, x, ev, color in bars:
            plane, full = ev[f"wasserstein2@{k}"] ** 2, ev[f"{FULL}@{k}"] ** 2
            ax.bar(x, plane, width * 0.95, color=color)
            ax.bar(x, full - plane, width * 0.95, bottom=plane, color=color, alpha=0.35)
        ax.set_xticks([-1, *range(len(sigmas))], ["clean", *sigmas])
        ax.set_xlabel("noise level σ")
        ax.set_title(title(k), fontweight="normal")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("W2² in 64-D")
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUES[ARMS.index(label)]) for label in arms]
    handles += [plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.35)]
    axes[-1].legend(handles, [*arms.values(), "solid: in the plane", "light: off the plane"],
                    loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.savefig(out / f"{name}-breakdown.png")
    plt.close(fig)


def scatter(ax, x, color, lim=2.3, size=1.0):
    ax.scatter([p[0] for p in x], [p[1] for p in x], s=size, linewidths=0, alpha=0.75, color=color,
               rasterized=True)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[], aspect="equal")
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(True)


def fig_compare_at(out, sets, arms, sigma, name, hist_step=8):
    """Arms at one noise level: W2 in 64-D against sampling steps (clean iMF dashed), each
    arm's distance from the plane at `hist_step` steps, and its samples in the plane at every
    step count, one row per arm."""
    evals, samples = sets[sigma]
    fig = plt.figure(figsize=(16, 3.6))
    left, mid, right = fig.subfigures(1, 3, width_ratios=[1, 1, 2.3])
    ax = left.subplots()
    ax.plot(STEPS, [evals[ARMS[0]][f"{FULL}@{k}"] for k in STEPS], color=MUTED, linestyle="--",
            linewidth=1.2, label="iMF, clean data")
    for label, short in arms.items():
        ax.plot(STEPS, [evals[label][f"{FULL}@{k}"] for k in STEPS], color=HUES[ARMS.index(label)],
                marker="o", label=short)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS, [str(k) for k in STEPS])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel("sampling steps")
    ax.set_ylabel(W2_LABEL[FULL])
    ax.legend(loc="best", fontsize=9)

    ax = mid.subplots()
    top = max(max(samples[label]["off"][str(hist_step)]) for label in [ARMS[0], *arms])
    bins = [i * top / 50 for i in range(51)]
    ax.hist(samples[ARMS[0]]["off"][str(hist_step)], bins=bins, histtype="step", color=MUTED, linewidth=1.5,
            linestyle="--", label="iMF, clean data")
    for label, short in arms.items():
        ax.hist(samples[label]["off"][str(hist_step)], bins=bins, histtype="step", linewidth=1.8,
                color=HUES[ARMS.index(label)], label=short)
    ax.set_yscale("log")
    ax.set_xlabel(f"distance from the plane at {hist_step} steps")
    ax.set_ylabel("samples (log scale)")
    ax.legend(loc="upper right", fontsize=9)

    axes = right.subplots(len(arms), len(STEPS), squeeze=False)
    for i, (label, short) in enumerate(arms.items()):
        for j, k in enumerate(STEPS):
            a = axes[i, j]
            scatter(a, samples[label][str(k)][:2048], HUES[ARMS.index(label)], size=0.8)
            a.text(0.04, 0.04, f"{evals[label][f'{FULL}@{k}']:.3f}", transform=a.transAxes, fontsize=7.5, color=INK)
            if i == 0:
                a.set_title(f"{k} step{'s' if k > 1 else ''}", fontsize=9, loc="center", fontweight="normal", color=INK)
            if j == 0:
                a.set_ylabel(short, fontsize=9, color=INK)
    fig.suptitle(f"σ = {sigma}", x=0.01, ha="left", fontsize=12, fontweight="normal")
    fig.savefig(out / f"{name}-sigma-{sigma}.png")
    plt.close(fig)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in SETS.items():
        if not all((LOGS / j / "eval.txt").exists() for j in spec["jobs"]):
            print(f"{name}: inputs missing, skipped")
            continue
        evals, samples = load(spec)
        evals = {label: evals[label] for label in spec["runs"]}
        fig_steps(name, spec, evals, out)
        fig_samples(name, spec, evals, samples, out)
        print(f"{name}: wrote {name}-steps.png, -samples.png")
    if all((LOGS / job / "eval.txt").exists() for job, _ in SWEEP.values()):
        sets = sweep()
        fig_sigma(out, sets, {label: label for label in ARMS[1:]}, "sweep-sigma")
        fig_sigma(out, sets, {label: label for label in ARMS[1:]}, "sweep-sigma", metric="wasserstein2")
        gaussian = {"2 naive, Gaussian": "iMF", "3 ambient, Gaussian": "ambient iMF"}
        fig_sigma(out, sets, gaussian, "gaussian-imf-vs-ambient")
        fig_breakdown(out, sets, gaussian, "gaussian-imf-vs-ambient")
        uniform = {"5 naive, uniform": "iMF", "4 ambient, uniform": "ambient iMF"}
        fig_sigma(out, sets, uniform, "uniform-imf-vs-ambient")
        fig_breakdown(out, sets, uniform, "uniform-imf-vs-ambient")
        for sigma in SWEEP:
            fig_compare_at(out, sets, gaussian, sigma, "gaussian-imf-vs-ambient")
            fig_compare_at(out, sets, uniform, sigma, "uniform-imf-vs-ambient")
        print("wrote sweep-sigma-w2.png, gaussian-/uniform-imf-vs-ambient-*.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
