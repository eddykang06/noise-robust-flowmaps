"""True vs generated figures for the 2-D toys.

One row per run: the training data (noisy samples in orange), the true
distribution (fresh clean samples), the generated samples at each sampler step
count, and the metric over training against its floor, the score of two
independent clean samples.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIM = 3.0  # toys have unit variance per coordinate


def scatter(ax, x, title, color="tab:blue"):
    x = x.cpu()
    ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.4, color=color, linewidths=0)
    ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[], title=title)


def row(axes, run: dict):
    """`run` holds `label`, `data` (Batch), `true` and `generated` ({steps: x})
    samples, `records` (its runs-file records), `metric` and `floor`."""
    name, data, last = run["metric"], run["data"], run["records"][-1]["metrics"]
    noisy = data.sigma > 0
    title = f"training data ({int(noisy.sum())} noisy)"
    scatter(axes[0], data.x[noisy], title, color="tab:orange")
    scatter(axes[0], data.x[~noisy], title)
    scatter(axes[1], run["true"], "true")
    for ax, (steps, x) in zip(axes[2:], run["generated"].items()):
        scatter(ax, x, f"generated, {steps}-step\n{name} {last[f'{name}@{steps}']:.3f}")
    ax = axes[-1]
    for key in last:
        ax.plot([r["step"] for r in run["records"]], [r["metrics"][key] for r in run["records"]],
                marker="o", label=key)
    ax.axhline(run["floor"], color="gray", linestyle="--", label=f"floor {run['floor']:.3f}")
    ax.set(yscale="log", xlabel="training step", title=name)
    ax.legend(fontsize=7)
    axes[0].set_ylabel(run["label"], fontsize=8)


def save(runs: list[dict], out):
    ncols = 3 + len(runs[0]["generated"])
    fig, axes = plt.subplots(len(runs), ncols, figsize=(3 * ncols, 3.2 * len(runs)), squeeze=False)
    for axes_row, run in zip(axes, runs):
        row(axes_row, run)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
