"""Sample-quality metrics `metric(samples, reference) -> float`; lower is better."""

import torch
from scipy.optimize import linear_sum_assignment


def wasserstein2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact W2 between two empirical distributions of equal size, on flattened
    samples. With uniform weights the optimal plan is a permutation, so it is the
    linear assignment on squared Euclidean distances."""
    assert len(a) == len(b), "W2 by assignment needs equal sample sizes"
    a, b = a.flatten(1).double().cpu(), b.flatten(1).double().cpu()
    cost = torch.cdist(a, b).square().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean() ** 0.5)


METRICS = {"wasserstein2": wasserstein2}
