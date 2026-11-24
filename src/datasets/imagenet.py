"""ImageNet (ILSVRC-2012) at 256 x 256 with its 1,000 class labels, as SD-VAE latents
`[N, 4, 32, 32]` (the iMF and DiT setup), memory-mapped from disk.

`prepare(src, split)` center-crops each image to 256 (ADM preprocessing), encodes it with
the SD-VAE (ft-EMA) and stores the latent mean times 0.18215; the train split also stores
each image flipped. Noise is added to the latents. A corrupted training set is computed once,
in chunks, and cached under <root>/imagenet/observed/, so each sample keeps one fixed noise
draw. `data.n` takes the first n of a seeded permutation (held in memory); null for all.
"""

import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np
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


@torch.no_grad()
def prepare(src: Path, split: str):
    """Encodes `src/<class folder>/<image>` (classes indexed in sorted folder order) into
    <root>/imagenet/{split}_latents.npy and {split}_labels.npy."""
    classes = sorted(p.name for p in src.iterdir() if p.is_dir())
    files = [(f, c) for c, name in enumerate(classes) for f in sorted((src / name).iterdir())]
    flips = 2 if split == "train" else 1
    out = root() / "imagenet"
    out.mkdir(parents=True, exist_ok=True)
    latents = np.lib.format.open_memmap(out / f"{split}_latents.npy", "w+", np.float32, (flips * len(files), 4, 32, 32))
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        x = torch.from_numpy(np.stack([center_crop(Image.open(f)) for f, _ in chunk])).to(DEVICE)
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
        for k in range(flips):
            z = vae().encode(x.flip(3) if k else x).latent_dist.mean * SCALE
            latents[k * len(files) + i:k * len(files) + i + len(chunk)] = z.cpu().numpy()
    latents.flush()
    np.save(out / f"{split}_labels.npy", np.tile([c for _, c in files], flips).astype(np.int64))


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
    val = load("val")
    return val[torch.randperm(len(val), generator=gen)[:n]]


@torch.no_grad()
def to_images(x: torch.Tensor) -> torch.Tensor:
    images = [vae().decode(z.to(DEVICE).float() / SCALE).sample for z in x.split(BATCH)]
    return ((torch.cat(images).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def figure(cfg: Config, data: Batch, ref: Batch, generated: dict, path):
    image_grid(to_images(ref.x[:8]), {k: to_images(x[:8]) for k, x in generated.items()}, path)
