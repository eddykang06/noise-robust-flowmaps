"""ImageNet (ILSVRC-2012) at 256 x 256 with its 1,000 class labels, in the latent space of the
SD-VAE ft-MSE, as iMF trains it (github.com/Lyy-iiis/imeanflow: utils/data_util.py, vae_util.py).

Each image is stored as its VAE posterior: the mean, normalized per channel by iMF's constants,
(mean - MEAN) / STD, as `x`, and the posterior std in the same units as `x_std`. Training draws
a fresh latent x + x_std * eps every step (`Batch.resample`), as iMF samples the posterior; there
are no flips (iMF has none). Images were ADM center-cropped to 256 before encoding.

The posteriors come precomputed from ChocolateDave/imagenet-256-sd-vae-latents (iMF's preparation:
Flax SD-VAE ft-MSE, ADM crop, unscaled mean and std; ImageNet-derived, so ImageNet's terms of use
apply). `download()` fetches it and `assemble(split)` writes the memory-mapped files below.

Noise is added to the normalized posterior mean, once per sample, and cached under
<root>/imagenet/observed/, so each noisy sample is a fresh latent plus its own fixed noise.
`data.n` takes the first n of a seeded permutation (held in memory); null for all. The reference
for metrics is class-balanced posterior means of the validation split.
"""

import hashlib
import json
from dataclasses import replace
from functools import cache

import numpy as np
import pyarrow.parquet as pq
import torch

from src.config import Config
from src.data import DEVICE, Batch
from src.datasets import root
from src.plots import image_grid

ON_DEVICE = False
VAE = "stabilityai/sd-vae-ft-mse"
CACHE = "ChocolateDave/imagenet-256-sd-vae-latents"
SPLITS = {"train": "train", "val": "validation"}
MEAN = np.array([0.86488, -0.27787343, 0.21616915, 0.3738409], np.float32).reshape(4, 1, 1)  # iMF's
STD = np.array([4.85503674, 5.31922414, 3.93725398, 3.9870003], np.float32).reshape(4, 1, 1)
CHUNK = 4096   # samples per chunk when caching a corrupted training set
BATCH = 16     # latents per VAE decoder pass


@cache
def vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE, cache_dir=root() / "weights").eval().to(DEVICE)


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(CACHE, repo_type="dataset", local_dir=root() / "imagenet" / "posterior",
                      allow_patterns=["data/*", "metadata/*"])


def assemble(split: str):
    """The cached posteriors of `split`, in shard order -> <root>/imagenet/{split}_x.npy,
    {split}_x_std.npy (normalized mean and std) and {split}_labels.npy (written last)."""
    files = sorted((root() / "imagenet" / "posterior" / "data").glob(f"{SPLITS[split]}-*.parquet"))
    assert files, f"no cached {split} posteriors: run `prepare_data imagenet-download`"
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out = root() / "imagenet"
    x, x_std = (np.lib.format.open_memmap(out / f"{split}_{name}.npy", "w+", np.float32, (n, 4, 32, 32))
                for name in ("x", "x_std"))
    labels, start = [], 0
    for f in files:
        table = pq.read_table(f, columns=["image", "label"])
        posterior = table.column("image").combine_chunks().flatten().flatten().flatten().to_numpy()
        posterior = posterior.reshape(-1, 8, 32, 32)
        x[start:start + len(posterior)] = (posterior[:, :4] - MEAN) / STD
        x_std[start:start + len(posterior)] = posterior[:, 4:] / STD
        labels.append(table.column("label").to_numpy())
        start += len(posterior)
    x.flush()
    x_std.flush()
    np.save(out / f"{split}_labels.npy", np.concatenate(labels).astype(np.int64))


def load(split: str) -> Batch:
    d = root() / "imagenet"
    x, x_std = (torch.from_numpy(np.load(d / f"{split}_{name}.npy", mmap_mode="c")) for name in ("x", "x_std"))
    y = torch.from_numpy(np.load(d / f"{split}_labels.npy"))
    return Batch(x=x, sigma=torch.zeros(len(y)), cond=y, x_std=x_std)


def build(cfg: Config, corrupt, gen: torch.Generator) -> Batch:
    clean = load("train")
    if cfg.data["n"] is not None:
        clean = clean[torch.randperm(len(clean), generator=gen)[:cfg.data["n"]]]
    if cfg.corruption["name"] == "none":
        return clean
    key = hashlib.sha1(json.dumps([cfg.seed, cfg.data, cfg.corruption], sort_keys=True).encode()).hexdigest()[:12]
    path = root() / "imagenet" / "observed" / key
    if not (path / "sigma.npy").exists():
        path.mkdir(parents=True, exist_ok=True)
        x = np.lib.format.open_memmap(path / "x.npy", "w+", np.float32, tuple(clean.x.shape))
        sigma = []
        for i in range(0, len(clean), CHUNK):
            observed = corrupt(Batch(x=clean.x[i:i + CHUNK], sigma=clean.sigma[i:i + CHUNK]))
            x[i:i + len(observed)] = observed.x.numpy()
            sigma.append(observed.sigma)
        x.flush()
        np.save(path / "sigma.npy", torch.cat(sigma).numpy())  # written last: marks the cache complete
    x = torch.from_numpy(np.load(path / "x.npy", mmap_mode="c"))
    return replace(clean, x=x, sigma=torch.from_numpy(np.load(path / "sigma.npy")))


def reference(cfg: Config, n: int, gen: torch.Generator) -> Batch:
    """`n` seeded validation posterior means, the same number from each class (balanced, as
    iMF's FID samples), grouped by class."""
    classes = cfg.model["num_classes"]
    assert n % classes == 0, f"metric.n_samples must be a multiple of the {classes} classes"
    val = load("val")
    order = torch.randperm(len(val), generator=gen)
    labels = val.cond[order]
    keep = torch.cat([order[labels == c][:n // classes] for c in range(classes)])
    assert len(keep) == n, "some class has too few validation latents"
    return replace(val[keep], x_std=None)


@torch.no_grad()
def to_images(x: torch.Tensor) -> torch.Tensor:
    """Normalized latents -> uint8 images, undoing iMF's normalization before decoding."""
    mean, std = (torch.from_numpy(a).to(DEVICE) for a in (MEAN, STD))
    images = [vae().decode(z.to(DEVICE).float() * std + mean).sample for z in x.split(BATCH)]
    return ((torch.cat(images).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def figure(cfg: Config, data: Batch, ref: Batch, generated: dict, path):
    image_grid(to_images(ref.x[:8]), {k: to_images(x[:8]) for k, x in generated.items()}, path)
