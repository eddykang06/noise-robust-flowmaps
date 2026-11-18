"""Behaviour fingerprint of a checkout of the shared pipeline, for comparing two versions.

    python <this> dump <out.pt>          # run from the checkout's root
    python <this> compare <a.pt> <b.pt>  # exits non-zero unless every entry is bitwise equal

Covers the training sets and references of several toy configs, the iMF loss, its logs
and gradients (toy MLP and a small imfDiT on images, guided and unguided, norm_p 0 and 1),
the sampler, and, with a GPU, short training runs: their run ids and metric records,
`evaluate` on the saved checkpoint and `scripts.sample` output.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())  # the checkout being fingerprinted

CONFIGS = {
    "clean": [],
    "spiral": ["data.name=spiral"],
    "corrupt": ["corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}"],
    "clean-only": ["corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}", "data.keep_noisy=false"],
}
TRAIN = ["train.steps=600", "train.eval_every=300", "train.lr=1.0e-3", "train.ema=0.99"]


def loss_entries(out, name, net, batch, **kw):
    from src.objectives import imf
    torch.manual_seed(7)
    value, logs = imf.loss(net, batch, P_mean=-0.4, P_std=1.0, data_proportion=0.5, cfg_beta=1.0,
                           class_dropout_prob=0.1, norm_eps=0.01, **kw)
    value.backward()
    out[f"{name}/loss"] = value.detach()
    out.update({f"{name}/log/{k}": v for k, v in logs.items()})
    # Parameters the loss does not reach (e.g. imfDiT's unused embedders) have no gradient.
    out.update({f"{name}/grad/{k}": None if p.grad is None else p.grad.clone()
                for k, p in net.named_parameters()})


def dump(path):
    from src.config import load
    from src.data import Batch
    from src.models import MLP, imfDiT
    from src.sampling import imf as sample
    from src.training import build_dataset, evaluate, load_run, reference, train

    out = {}
    for name, over in CONFIGS.items():
        cfg = load("configs/toy.yaml", ["slug=parity", *over])
        out[f"{name}/run_id"] = cfg.run_id
        for part, batch in [("data", build_dataset(cfg)), ("ref", reference(cfg))]:
            out[f"{name}/{part}/x"], out[f"{name}/{part}/sigma"] = batch.x, batch.sigma

    def mlp(k):
        torch.manual_seed(0)
        net = MLP(dim=2, hidden=64, depth=3, aux_head_depth=1, num_classes=k)
        for p in net.parameters():
            torch.nn.init.normal_(p, std=0.3)
        return net

    x = torch.randn(64, 2, generator=torch.Generator().manual_seed(3))
    labels = torch.arange(64) % 3
    for p in [0.0, 1.0]:
        loss_entries(out, f"mlp-unguided-p{p}", mlp(0), Batch(x=x, sigma=torch.zeros(64)),
                     norm_p=p, guidance=False)
        loss_entries(out, f"mlp-guided-p{p}", mlp(3), Batch(x=x, sigma=torch.zeros(64), cond=labels),
                     norm_p=p, guidance=True)
    torch.manual_seed(0)
    dit = imfDiT.imfDiT(input_size=8, patch_size=2, in_channels=1, hidden_size=64, depth=2,
                        num_heads=4, num_classes=3, aux_head_depth=1)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    images = torch.randn(4, 1, 8, 8, generator=torch.Generator().manual_seed(4))
    loss_entries(out, "dit-guided", dit, Batch(x=images, sigma=torch.zeros(4), cond=torch.arange(4) % 3),
                 norm_p=1.0, guidance=True)
    with torch.no_grad():
        for steps in [1, 2, 5]:
            out[f"sample/{steps}"] = sample(mlp(3), x, steps, labels, omega=2.0, t_min=0.2, t_max=0.8)

    if torch.cuda.is_available():
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["clean", "corrupt", "clean-only"]:
                cfg = load("configs/toy.yaml", ["slug=parity", f"out_root={tmp}", f"runs_dir={tmp}/{name}",
                                                *CONFIGS[name], *TRAIN])
                train(cfg)
                records = [json.loads(line) for line in (Path(tmp) / name / "parity.jsonl").read_text().splitlines()]
                out[f"train/{name}/records"] = [(r["run_id"], r["step"], r["metrics"]) for r in records]
                run_dir = Path(tmp) / "parity" / cfg.run_id
                cfg_run, net = load_run(run_dir)
                out[f"train/{name}/evaluate"] = evaluate(net, cfg_run, reference(cfg_run).to("cuda"))
                subprocess.run([sys.executable, "-m", "scripts.sample", str(run_dir), "--n", "512",
                                "--steps", "2", "--seed", "0"], check=True)
                out[f"train/{name}/sample"] = torch.load(run_dir / "samples-2step.pt")
    torch.save(out, path)
    print(f"wrote {len(out)} entries to {path}")


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    keys = sorted(set(a) | set(b))
    bad = [k for k in keys if k not in a or k not in b or not equal(a[k], b[k])]
    print(f"{len(keys) - len(bad)} of {len(keys)} entries bitwise equal")
    for k in bad:
        print("DIFFERS:", k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    {"dump": lambda: dump(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
