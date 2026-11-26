"""Checks the ImageNet posterior cache (ChocolateDave/imagenet-256-sd-vae-latents) against the raw
ILSVRC/imagenet-1k Parquet it was made from: same rows and labels, and our own PyTorch SD-VAE
ft-MSE encoding of 16 validation images (MeanFlow's ADM center crop) matches its mean and std;
also that its per-channel latent statistics are near iMF's normalization constants.
Usage: python verify_latent_cache.py   (NRFM_DATA set; run from main's tree; CPU is enough)."""
import io

import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers import AutoencoderKL
from PIL import Image

from src.datasets import imagenet, root


def center_crop(image, size=256):
    """ADM / MeanFlow center_crop_arr."""
    while min(*image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)
    scale = size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)
    arr = np.array(image)
    top, left = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return arr[top:top + size, left:left + size]


d = root() / "imagenet"
raw = pq.read_table(d / "hf" / "data" / "validation-00000-of-00014.parquet")
cached = pq.read_table(d / "posterior" / "data" / "validation-00000-of-00014.parquet")
assert len(raw) == len(cached) and np.array_equal(raw.column("label").to_numpy(), cached.column("label").to_numpy())
assert np.array_equal(cached.column("source_index").to_numpy(), np.arange(len(raw)))  # shard 0: row i
print(f"rows and labels match: {len(raw)} rows")

posterior = cached.column("image").combine_chunks().flatten().flatten().flatten().to_numpy().reshape(-1, 8, 32, 32)
rows = np.linspace(0, len(raw) - 1, 16).astype(int)
images = np.stack([center_crop(Image.open(io.BytesIO(raw.column("image")[int(i)].as_py()["bytes"])).convert("RGB"))
                   for i in rows])
x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1
vae = AutoencoderKL.from_pretrained(imagenet.VAE, cache_dir=root() / "weights").eval()
with torch.no_grad():
    dist = vae.encode(x).latent_dist
ours_mean, ours_std = dist.mean.numpy(), dist.std.numpy()
theirs_mean, theirs_std = posterior[rows, :4], posterior[rows, 4:]
rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
print(f"mean: max |ours - cache| = {np.abs(ours_mean - theirs_mean).max():.4f} "
      f"(relative to max |cache| {rel(ours_mean, theirs_mean):.2e}); corr {np.corrcoef(ours_mean.ravel(), theirs_mean.ravel())[0, 1]:.6f}")
print(f"std:  max |ours - cache| = {np.abs(ours_std - theirs_std).max():.5f} "
      f"(relative {rel(ours_std, theirs_std):.2e}); typical std {np.median(theirs_std):.4f}")

latents = posterior[:, :4] + posterior[:, 4:] * np.random.default_rng(0).standard_normal(posterior[:, 4:].shape, np.float32)
print("per-channel mean of sampled latents, shard 0:", latents.mean((0, 2, 3)).round(3), "iMF:", imagenet.MEAN.ravel().round(3))
print("per-channel std  of sampled latents, shard 0:", latents.std((0, 2, 3)).round(3), "iMF:", imagenet.STD.ravel().round(3))
