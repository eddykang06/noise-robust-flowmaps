"""CPU smoke test of main's real-data pipeline, run from the tree's root with NRFM_DATA set:
the image metrics on real MNIST against references whose answer is known, and a fake ImageNet
(smooth random images in class folders) through the SD-VAE and Inception.

    python <this file> <work dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import load
from src.data import Batch
from src.datasets import imagenet, mnist, root
from src.metrics import image_scores
from src.training import reference

work = Path(sys.argv[1])
show = lambda name, scores: print(name, {k: round(v, 4) for k, v in scores.items()}, flush=True)

# MNIST: the reference itself (FID 0), other real digits (small FID), uniform noise (large FID).
cfg = load("configs/mnist.yaml", ["slug=smoke", "metric.n_samples=2000"])
ref = reference(cfg)
train_x, _ = mnist.load("train")
show("mnist", image_scores(cfg, {1: ref.x, 2: train_x[:2000], 3: torch.rand_like(ref.x) * 2 - 1}, ref))

# Fake ImageNet: low-frequency random images, so the VAE can reconstruct them.
rng = np.random.default_rng(0)
for split, n in [("train", 20), ("val", 8)]:
    for c in ["n01", "n02", "n03"]:
        (work / "fake" / split / c).mkdir(parents=True)
        for i in range(n):
            h, w = rng.integers(260, 600, 2)
            field = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC)
            field.save(work / "fake" / split / c / f"{i}.JPEG", quality=95)
imagenet.BATCH = 4  # VAE passes on CPU
for split in ["train", "val"]:
    imagenet.prepare(work / "fake" / split, split)
val = imagenet.load("val")
print("imagenet latents", tuple(imagenet.load("train").x.shape), tuple(val.x.shape), "labels", val.cond.tolist())
files = sorted((work / "fake" / "val").glob("*/*.JPEG"))
crops = torch.from_numpy(np.stack([imagenet.center_crop(Image.open(f)) for f in files])).permute(0, 3, 1, 2)
mse = (imagenet.to_images(val.x).cpu().float() - crops.float()).square().mean()
print(f"imagenet VAE reconstruction PSNR {10 * torch.log10(255 ** 2 / mse):.1f} dB", flush=True)
cfg = load("configs/imagenet.yaml", ["slug=smoke", "metric.n_samples=24"])
show("imagenet", image_scores(cfg, {1: val.x, 2: torch.randn_like(val.x)}, val))
