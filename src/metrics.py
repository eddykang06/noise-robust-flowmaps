"""Sample-quality metrics `metric(samples, reference) -> float`; lower is better."""

import torch


def energy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """2 E|X - Y| - E|X - X'| - E|Y - Y'| (V-statistic), on flattened samples."""
    a, b = a.flatten(1).float(), b.flatten(1).float()
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()).item()


METRICS = {"energy_distance": energy_distance}
