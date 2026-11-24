"""MNIST digits with their labels, padded from 28 x 28 to 32 x 32 (for patch size 2 and 4)
and scaled to [-1, 1]: `[N, 1, 32, 32]`. Training uses the train split (the first
`data.n` of a seeded permutation; null for all 60,000), the reference the test split.
The image metrics use the features of a small classifier trained on the train split
(`classify`), as there is no standard MNIST feature network."""

import gzip
import urllib.request
from functools import cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import Config
from src.data import DEVICE, Batch
from src.datasets import root
from src.plots import image_grid

ON_DEVICE = True
URL = "https://ossci-datasets.s3.amazonaws.com/mnist/{}-ubyte.gz"  # the torchvision mirror
FILES = {"train": ("train-images-idx3", "train-labels-idx1"), "test": ("t10k-images-idx3", "t10k-labels-idx1")}


def download():
    """The four IDX files -> <root>/mnist/{train,test}.npz with uint8 `images` and `labels`."""
    (root() / "mnist").mkdir(parents=True, exist_ok=True)
    for split, (images, labels) in FILES.items():
        read = lambda name, offset: np.frombuffer(gzip.decompress(urllib.request.urlopen(URL.format(name)).read()),
                                                   np.uint8, offset=offset)
        np.savez(root() / "mnist" / f"{split}.npz", images=read(images, 16).reshape(-1, 28, 28),
                 labels=read(labels, 8))


def load(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    files = np.load(root() / "mnist" / f"{split}.npz")
    x = F.pad(torch.from_numpy(files["images"]).float()[:, None] / 127.5 - 1, (2, 2, 2, 2), value=-1.0)
    return x, torch.from_numpy(files["labels"]).long()


def subset(split: str, n, gen: torch.Generator) -> Batch:
    x, y = load(split)
    idx = torch.randperm(len(y), generator=gen)[:n]
    return Batch(x=x[idx], sigma=torch.zeros(len(idx)), cond=y[idx])


def build(cfg: Config, corrupt, gen: torch.Generator) -> Batch:
    return corrupt(subset("train", cfg.data["n"], gen))


def reference(cfg: Config, n: int, gen: torch.Generator) -> Batch:
    return subset("test", n, gen)


def to_images(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).expand(-1, 3, -1, -1)


def figure(cfg: Config, data: Batch, ref: Batch, generated: dict, path):
    image_grid(to_images(ref.x[:8]), {k: to_images(x[:8]) for k, x in generated.items()}, path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.features(x))


def train_classifier(epochs: int = 5, seed: int = 0) -> float:
    """Trains the metric classifier on the train split; saves it and returns test accuracy."""
    torch.manual_seed(seed)
    net, (x, y) = Classifier().to(DEVICE), load("train")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(y)).split(256):
            loss = F.cross_entropy(net(x[idx].to(DEVICE)), y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(net.state_dict(), root() / "mnist" / "classifier.pt")
    x, y = load("test")
    with torch.no_grad():
        return (net.eval()(x.to(DEVICE)).argmax(1).cpu() == y).float().mean().item()


@cache
def classifier() -> Classifier:
    net = Classifier()
    net.load_state_dict(torch.load(root() / "mnist" / "classifier.pt", map_location="cpu"))
    return net.eval().to(DEVICE)


@torch.no_grad()
def classify(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Features and class probabilities of uint8 images `[B, 3, 32, 32]`."""
    net, feats, probs = classifier(), [], []
    for chunk in images.split(1000):
        f = net.features(chunk[:, :1].to(DEVICE).float() / 127.5 - 1)
        feats.append(f)
        probs.append(net.head(f).softmax(1))
    return torch.cat(feats), torch.cat(probs)
