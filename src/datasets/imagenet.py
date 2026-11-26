"""ImageNet (ILSVRC-2012) at 256 x 256 with its 1,000 class labels, as SD-VAE latents
`[N, 4, 32, 32]` (the iMF and DiT setup), memory-mapped from disk.

Prepared in three steps (scripts/prepare_data.py): `download()` fetches the Parquet shards of
Hugging Face's ILSVRC/imagenet-1k (gated: accept its terms and `hf auth login` first);
`encode(split, first, last)` center-crops each image of a range of shards to 256 (ADM
preprocessing), encodes it with the SD-VAE (ft-EMA) and stores the latent mean times 0.18215,
the train split also each image flipped, one file per shard, resumably; `assemble(split)`
concatenates the shards into the memory-mapped training files. Noise is added to the latents. A corrupted training set is computed once,
in chunks, and cached under <root>/imagenet/observed/, so each sample keeps one fixed noise
draw. `data.n` takes the first n of a seeded permutation (held in memory); null for all.
The reference for metrics is class-balanced.
"""

import hashlib
import io
import json
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from src.config import Config
from src.data import DEVICE, Batch
from src.datasets import root
from src.plots import image_grid

ON_DEVICE = False
SCALE = 0.18215  # SD-VAE latent scale
VAE = "stabilityai/sd-vae-ft-ema"
CHUNK = 4096   # samples per chunk when caching a corrupted training set
BATCH = 16     # images per VAE pass
HF = "ILSVRC/imagenet-1k"
HF_SPLIT = {"train": "train", "val": "validation"}


@cache
def vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE, cache_dir=root() / "weights").eval().to(DEVICE)


def center_crop(image: Image.Image, size: int = 256) -> np.ndarray:
    """ADM's center_crop_arr: halve while large, resize the short side to `size`, crop."""
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(s // 2 for s in image.size), resample=Image.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(s * scale) for s in image.size), resample=Image.BICUBIC)
    arr = np.asarray(image.convert("RGB"))
    top, left = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return arr[top:top + size, left:left + size]


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(HF, repo_type="dataset", local_dir=root() / "imagenet" / "hf",
                      allow_patterns=[f"data/{name}-*" for name in HF_SPLIT.values()])


def shards(split: str) -> list[Path]:
    return sorted((root() / "imagenet" / "hf" / "data").glob(f"{HF_SPLIT[split]}-*.parquet"))


def encoded(split: str, k: int) -> Path:
    return root() / "imagenet" / "shards" / f"{split}-{k:05d}.npz"


@torch.no_grad()
def encode(split: str, first: int, last: int):
    """Encodes Parquet shards [first, last) of `split` into one file each, skipping shards
    already encoded. A file is written whole, so an interrupted job loses only its current shard."""
    flips = 2 if split == "train" else 1
    for k, path in enumerate(shards(split)[first:last], first):
        if encoded(split, k).exists():
            continue
        table = pq.read_table(path)
        images = [row["bytes"] for row in table.column("image").to_pylist()]
        latents = np.empty((flips, len(images), 4, 32, 32), np.float32)
        for i in range(0, len(images), BATCH):
            x = np.stack([center_crop(Image.open(io.BytesIO(b))) for b in images[i:i + BATCH]])
            x = torch.from_numpy(x).to(DEVICE).permute(0, 3, 1, 2).float() / 127.5 - 1
            for f in range(flips):
                latents[f, i:i + len(x)] = (vae().encode(x.flip(3) if f else x).latent_dist.mean * SCALE).cpu().numpy()
        encoded(split, k).parent.mkdir(parents=True, exist_ok=True)
        tmp = encoded(split, k).with_suffix(".tmp.npz")
        np.savez(tmp, latents=latents, labels=table.column("label").to_numpy())
        tmp.rename(encoded(split, k))


def assemble(split: str):
    """All encoded shards of `split` -> <root>/imagenet/{split}_latents.npy (every image in shard
    order, then for train every flip in the same order) and {split}_labels.npy. The shard files
    can be deleted afterwards."""
    files = [encoded(split, k) for k in range(len(shards(split)))]
    missing = [f.name for f in files if not f.exists()]
    assert files and not missing, f"not encoded yet: {missing or split}"
    labels = [np.load(f)["labels"] for f in files]
    n, flips = sum(map(len, labels)), 2 if split == "train" else 1
    out = root() / "imagenet"
    latents = np.lib.format.open_memmap(out / f"{split}_latents.npy", "w+", np.float32, (flips * n, 4, 32, 32))
    start = 0
    for f, y in zip(files, labels):
        z = np.load(f)["latents"]
        for k in range(flips):
            latents[k * n + start:k * n + start + len(y)] = z[k]
        start += len(y)
    latents.flush()
    np.save(out / f"{split}_labels.npy", np.tile(np.concatenate(labels), flips).astype(np.int64))


def load(split: str) -> Batch:
    d = root() / "imagenet"
    x = torch.from_numpy(np.load(d / f"{split}_latents.npy", mmap_mode="c"))
    y = torch.from_numpy(np.load(d / f"{split}_labels.npy"))
    return Batch(x=x, sigma=torch.zeros(len(y)), cond=y)


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
            observed = corrupt(clean[i:i + CHUNK])
            x[i:i + len(observed)] = observed.x.numpy()
            sigma.append(observed.sigma)
        x.flush()
        np.save(path / "sigma.npy", torch.cat(sigma).numpy())  # written last: marks the cache complete
    x = torch.from_numpy(np.load(path / "x.npy", mmap_mode="c"))
    return Batch(x=x, sigma=torch.from_numpy(np.load(path / "sigma.npy")), cond=clean.cond)


def reference(cfg: Config, n: int, gen: torch.Generator) -> Batch:
    """`n` seeded validation latents, the same number from each class (balanced, as iMF's FID
    samples), grouped by class."""
    classes = cfg.model["num_classes"]
    assert n % classes == 0, f"metric.n_samples must be a multiple of the {classes} classes"
    val = load("val")
    order = torch.randperm(len(val), generator=gen)
    labels = val.cond[order]
    keep = torch.cat([order[labels == c][:n // classes] for c in range(classes)])
    assert len(keep) == n, "some class has too few validation latents"
    return val[keep]


@torch.no_grad()
def to_images(x: torch.Tensor) -> torch.Tensor:
    images = [vae().decode(z.to(DEVICE).float() / SCALE).sample for z in x.split(BATCH)]
    return ((torch.cat(images).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def figure(cfg: Config, data: Batch, ref: Batch, generated: dict, path):
    image_grid(to_images(ref.x[:8]), {k: to_images(x[:8]) for k, x in generated.items()}, path)
