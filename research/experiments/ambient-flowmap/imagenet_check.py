"""Checks one encoded ImageNet shard against its Parquet source: SD-VAE round-trip PSNR and a
labelled grid of decoded images (to see that labels name the pictured class).
Usage: imagenet_check.py <split> <shard> <figure.png>. Run from main's tree."""
import io
import runpy
import sys

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import imagenet, root

split, k, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
encoded = np.load(imagenet.encoded(split, k))
table = pq.read_table(imagenet.shards(split)[k])
assert np.array_equal(encoded["labels"], table.column("label").to_numpy())
idx = np.arange(0, len(table), len(table) // 12)[:12]
truth = np.stack([imagenet.center_crop(Image.open(io.BytesIO(table.column("image")[int(i)].as_py()["bytes"]))) for i in idx])
decoded = imagenet.to_images(torch.from_numpy(encoded["latents"][0, idx])).permute(0, 2, 3, 1).cpu().numpy()
mse = ((truth.astype(np.float64) - decoded) ** 2).mean()
print(f"{len(table)} images, latents {encoded['latents'].shape}, round-trip PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB")
names = list(runpy.run_path(str(root() / "imagenet" / "hf" / "classes.py"))["IMAGENET2012_CLASSES"].values())
fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for ax, image, i in zip(axes.flat, decoded, idx):
    ax.imshow(image); ax.set_title(names[encoded["labels"][i]][:28], fontsize=9); ax.axis("off")
fig.savefig(path, bbox_inches="tight", dpi=80)
