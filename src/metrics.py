"""Sample-quality metrics; lower is better unless noted.

`METRICS[metric.name](cfg, generated, ref) -> dict` scores generated samples, one tensor per
sampler step count, against the clean reference:
- `wasserstein2` (the spiral): exact W2 in all D dimensions (`wasserstein2_full`), in the
  plane (`wasserstein2`), and the RMS off the plane (`off_plane_rms`, 0 for clean data).
- `fid` (images): Frechet distance (`fid`), precision and recall (Kynkaanniemi et al. 2019,
  k = 3; higher is better), all on the features of `metric.features`, and the
  classifier's Inception-style score (`is`; higher is better). With `features: mnist` also the
  accuracy of the classifier on the intended labels (`accuracy`; higher is better).
"""

from functools import cache

import numpy as np
import torch
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from src.data import DEVICE
from src.embed import frame


def wasserstein2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact W2 between two empirical distributions of equal size, on flattened
    samples. With uniform weights the optimal plan is a permutation, so it is the
    linear assignment on squared Euclidean distances."""
    assert len(a) == len(b), "W2 by assignment needs equal sample sizes"
    a, b = a.flatten(1).double().cpu(), b.flatten(1).double().cpu()
    cost = torch.cdist(a, b).square().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean() ** 0.5)


def spiral_scores(cfg, generated: dict, ref) -> dict:
    name, plane = cfg.metric["name"], frame(ref.x)[:, :2]
    out = {}
    for steps, x in generated.items():
        framed = frame(x)
        out[f"{name}@{steps}"] = wasserstein2(framed[:, :2], plane)
        if x.shape[1] > 2:
            out[f"{name}_full@{steps}"] = wasserstein2(x, ref.x)
            out[f"off_plane_rms@{steps}"] = framed[:, 2:].square().mean().sqrt().item()
    return out


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frechet distance between Gaussians fitted to two feature sets."""
    a, b = a.double().cpu().numpy(), b.double().cpu().numpy()
    diff, cov_a, cov_b = a.mean(0) - b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(cov_a @ cov_b).real
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(root))


def precision_recall(real: torch.Tensor, fake: torch.Tensor, k: int = 3) -> tuple[float, float]:
    """Share of fake samples inside the real manifold (precision) and of real samples inside
    the fake one (recall); a manifold is the union of balls reaching each point's k-th neighbour."""
    def covered(points, support):
        radii = torch.cdist(support, support).kthvalue(k + 1, dim=1).values  # +1: itself
        return (torch.cdist(points, support) <= radii).any(1).float().mean().item()
    real, fake = real.float().to(DEVICE), fake.float().to(DEVICE)
    return covered(fake, real), covered(real, fake)


def class_score(probs: torch.Tensor) -> float:
    """exp E[KL(p(y|x) || p(y))]: the Inception score for Inception's classes."""
    probs = probs.double()
    return (probs * (probs.clamp_min(1e-12).log() - probs.mean(0).log())).sum(1).mean().exp().item()


@cache
def inception():
    """The FID Inception network as TorchScript (from StyleGAN2-ADA; scripts/prepare_data.py)."""
    from src.datasets import root
    return torch.jit.load(root() / "weights" / "inception-2015-12-05.pt", map_location=DEVICE).eval()


@torch.no_grad()
def inception_features(images: torch.Tensor):
    net, feats, probs = inception(), [], []
    for chunk in images.split(250):
        chunk = chunk.to(DEVICE)
        feats.append(net(chunk, return_features=True))
        probs.append(net(chunk, no_output_bias=True))
    return torch.cat(feats), torch.cat(probs)


def mnist_features(images: torch.Tensor):
    from src.datasets import mnist
    return mnist.classify(images)


FEATURES = {"inception": inception_features, "mnist": mnist_features}


def image_scores(cfg, generated: dict, ref) -> dict:
    from src.datasets import DATASETS
    to_images, features = DATASETS[cfg.data["name"]].to_images, FEATURES[cfg.metric["features"]]
    ref_feats, _ = features(to_images(ref.x))
    out = {}
    for steps, x in generated.items():
        feats, probs = features(to_images(x))
        out[f"fid@{steps}"] = frechet(feats, ref_feats)
        out[f"precision@{steps}"], out[f"recall@{steps}"] = precision_recall(ref_feats, feats)
        out[f"is@{steps}"] = class_score(probs)
        if cfg.metric["features"] == "mnist":
            out[f"accuracy@{steps}"] = (probs.argmax(1).cpu() == ref.cond.cpu()).float().mean().item()
    return out


METRICS = {"wasserstein2": spiral_scores, "fid": image_scores}
