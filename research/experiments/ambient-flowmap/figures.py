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
# The same five arms at sigma = 1.0 with 10x the training points (n = 100,000), trained by
# 60291354, 60291357, 60274565, 60313028, 60313029 and re-scored together by 60321685.
SETS["sigma-1.0-n100k"] = {
    "title": "Five arms, 75% clean, sigma = 1.0, n = 100,000",
    "runs": dict(zip(ARMS, [("imf-s42-24d83765", "c100k"), ("imf-s42-0d807dc0", "g100n100k"),
                            ("ambient_imf-s42-152f4102", "g100a100k"), ("ambient_imf-s42-e6c7074c", "u100a100k"),
                            ("imf-s42-8017d5d9", "u100n100k")])),
    "jobs": ["60321685"], "metric": FULL, "colors": HUES}

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
    sigmas = [float(s) for s in sets]
    clean = sets[next(iter(sets))][0][ARMS[0]]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.3 * len(steps), 3.2), sharey=True)
    for ax, k in zip(axes, steps):
        ax.axhline(clean[f"{metric}@{k}"], color=MUTED, linestyle="--", linewidth=1.2, label="iMF, clean data")
        for label, short in arms.items():
            ax.plot(sigmas, [sets[s][0][label][f"{metric}@{k}"] for s in sets], color=HUES[ARMS.index(label)],
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


PAIRS = {"Gaussian noise": ("2 naive, Gaussian", "3 ambient, Gaussian"),
         "Uniform noise": ("5 naive, uniform", "4 ambient, uniform")}
NAIVE, AMBIENT = HUES[1], HUES[2]


def fig_summary_w2(out, sets, steps=(1, 2, 4, 8)):
    """W2 in 64-D against sigma for every sigma in `sets`, one row per noise type and one column
    per step count; iMF vs ambient iMF, clean iMF dashed. Large type for reading on a page."""
    sigmas = [float(s) for s in sets]
    clean = sets[next(iter(sets))][0][ARMS[0]]
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "lines.linewidth": 2.4, "lines.markersize": 7}):
        fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 7.2), sharex=True, sharey=True)
        for row, (noise, (naive, ambient)) in zip(axes, PAIRS.items()):
            for ax, k in zip(row, steps):
                ax.axhline(clean[f"{FULL}@{k}"], color=MUTED, linestyle="--", linewidth=1.4)
                ax.plot(sigmas, [sets[s][0][naive][f"{FULL}@{k}"] for s in sets], color=NAIVE, marker="o")
                ax.plot(sigmas, [sets[s][0][ambient][f"{FULL}@{k}"] for s in sets], color=AMBIENT, marker="o")
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_ylim(0, None)
            row[0].set_ylabel(f"{noise}\nW2 in 64-D")
        for ax, k in zip(axes[0], steps):
            ax.set_title(title(k))
        for ax in axes[1]:
            ax.set_xlabel("noise level σ")
        handles = [plt.Line2D([], [], color=NAIVE, marker="o"), plt.Line2D([], [], color=AMBIENT, marker="o"),
                   plt.Line2D([], [], color=MUTED, linestyle="--")]
        fig.legend(handles, ["iMF (noisy points used as clean)", "ambient iMF", "iMF on clean data only"],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout()
        fig.savefig(out / "summary-w2.png")
        plt.close(fig)


def fig_summary_samples(out, sets, step=8):
    """Samples in the plane at `step` steps: one row per sigma in `sets`, the true spiral then iMF
    and ambient iMF for each noise type, W2 in 64-D in the corner."""
    cols = [("true spiral", None, MUTED)] + [
        (f"{who}\n{noise.split()[0]}", label, color)
        for noise, pair in PAIRS.items() for who, label, color in zip(("iMF", "ambient iMF"), pair, (NAIVE, AMBIENT))]
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(len(sets), len(cols), figsize=(2.3 * len(cols), 2.3 * len(sets)))
        for i, sigma in enumerate(sets):
            evals, samples = sets[sigma]
            for j, (name, label, color) in enumerate(cols):
                ax = axes[i, j]
                x = samples[ARMS[0]]["true"] if label is None else samples[label][str(step)]
                scatter(ax, x[:2048], color, lim=2.6, size=1.6)
                if label is not None:
                    ax.text(0.04, 0.04, f"{evals[label][f'{FULL}@{step}']:.2f}", transform=ax.transAxes,
                            fontsize=12, color=INK)
                if i == 0:
                    ax.set_title(name, fontsize=13, loc="center", fontweight="normal")
                if j == 0:
                    ax.set_ylabel(f"σ = {sigma}", fontsize=13, color=INK)
        fig.suptitle(f"Samples at {title(step)}", x=0.01, ha="left", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"summary-samples-{step}.png")
        plt.close(fig)


# Two-way comparisons for the results page: name -> {label: (arm, colour)}.
COMPARISONS = {
    "gaussian-imf-vs-ambient": {"iMF, Gaussian": ("2 naive, Gaussian", NAIVE),
                                "ambient iMF, Gaussian": ("3 ambient, Gaussian", AMBIENT)},
    "ambient-gaussian-vs-uniform": {"ambient iMF, Gaussian": ("3 ambient, Gaussian", AMBIENT),
                                    "ambient iMF, uniform": ("4 ambient, uniform", HUES[3])},
}


def fig_compare(out, sets, name, steps=(1, 2, 4, 8), step=4):
    """One two-way comparison: W2 in 64-D against sigma at each step count (clean iMF dashed),
    and the two arms' samples in the plane at `step` steps, one column per sigma."""
    pair, sigmas = COMPARISONS[name], [float(s) for s in sets]
    clean = sets[next(iter(sets))][0][ARMS[0]]
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "lines.linewidth": 2.4, "lines.markersize": 7}):
        fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 3.9), sharey=True)
        for ax, k in zip(axes, steps):
            ax.axhline(clean[f"{FULL}@{k}"], color=MUTED, linestyle="--", linewidth=1.4, label="iMF on clean data only")
            for label, (arm, color) in pair.items():
                ax.plot(sigmas, [sets[s][0][arm][f"{FULL}@{k}"] for s in sets], color=color, marker="o", label=label)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel("noise level σ")
            ax.set_title(title(k))
            ax.set_ylim(0, None)
        axes[0].set_ylabel("W2 in 64-D")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        fig.savefig(out / f"compare-{name}-w2.png")
        plt.close(fig)

        fig, axes = plt.subplots(len(pair), len(sets), figsize=(1.6 * len(sets), 1.75 * len(pair)))
        for i, (label, (arm, color)) in enumerate(pair.items()):
            for j, sigma in enumerate(sets):
                evals, samples = sets[sigma]
                ax = axes[i, j]
                scatter(ax, samples[arm][str(step)][:2048], color, lim=2.6, size=1.0)
                ax.text(0.05, 0.05, f"{evals[arm][f'{FULL}@{step}']:.2f}", transform=ax.transAxes, fontsize=11, color=INK)
                if i == 0:
                    ax.set_title(f"σ = {sigma}", fontsize=13, loc="center", fontweight="normal")
                if j == 0:
                    ax.set_ylabel(label.replace(", ", "\n"), fontsize=12, color=INK)
        fig.suptitle(f"Samples at {title(step)}", x=0.01, ha="left", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"compare-{name}-samples-{step}.png")
        plt.close(fig)


SIZES = {"n = 10,000": "sigma-1.0", "n = 100,000": "sigma-1.0-n100k"}
SHORT = {"1 clean": "iMF, clean data", "2 naive, Gaussian": "iMF, Gaussian", "3 ambient, Gaussian": "ambient iMF, Gaussian",
         "4 ambient, uniform": "ambient iMF, uniform", "5 naive, uniform": "iMF, uniform"}


def fig_data_size(out, steps=(1, 2, 4, 8), step=4):
    """sigma = 1.0 at both training-set sizes: W2 in 64-D against sampling steps for the five arms
    (one panel per size, shared axis), and samples in the plane at `step` steps, one row per size."""
    data = {size: load(SETS[name]) for size, name in SIZES.items()}
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "lines.linewidth": 2.4, "lines.markersize": 7}):
        fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4.4), sharey=True)
        for ax, (size, (evals, _)) in zip(axes, data.items()):
            for arm, color in zip(ARMS, HUES):
                ax.plot(steps, [evals[arm][f"{FULL}@{k}"] for k in steps], color=color, marker="o", label=SHORT[arm],
                        linestyle="--" if arm == ARMS[0] else "-")
            ax.set_xscale("log", base=2)
            ax.set_xticks(steps, [str(k) for k in steps])
            ax.minorticks_off()
            ax.set_xlabel("sampling steps")
            ax.set_title(f"σ = 1.0, {size} training points")
            ax.set_ylim(0, None)
        axes[0].set_ylabel("W2 in 64-D (lower is better)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "data-size-w2.png")
        plt.close(fig)

        fig, axes = plt.subplots(len(data), 1 + len(ARMS), figsize=(2.3 * (1 + len(ARMS)), 2.5 * len(data)))
        for i, (size, (evals, samples)) in enumerate(data.items()):
            for j, (arm, color) in enumerate([(None, MUTED), *zip(ARMS, HUES)]):
                ax = axes[i, j]
                scatter(ax, (samples[ARMS[0]]["true"] if arm is None else samples[arm][str(step)])[:2048], color,
                        lim=2.6, size=1.6)
                if arm is not None:
                    ax.text(0.04, 0.04, f"{evals[arm][f'{FULL}@{step}']:.2f}", transform=ax.transAxes, fontsize=12, color=INK)
                if i == 0:
                    ax.set_title("true spiral" if arm is None else SHORT[arm].replace(", ", "\n"), fontsize=12,
                                 loc="center", fontweight="normal")
                if j == 0:
                    ax.set_ylabel(size, fontsize=13, color=INK)
        fig.suptitle(f"Samples at {title(step)}, σ = 1.0", x=0.01, ha="left", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"data-size-samples-{step}.png")
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
        fig_summary_w2(out, sets)
        fig_summary_samples(out, sets)
        for name in COMPARISONS:
            fig_compare(out, sets, name)
        print("wrote sweep-sigma-w2.png, gaussian-/uniform-imf-vs-ambient-*.png, summary-*.png")
        fig_data_size(out)
        print("wrote data-size-w2.png, data-size-samples-4.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
