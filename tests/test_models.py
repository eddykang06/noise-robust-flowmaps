import hashlib
from pathlib import Path

import pytest
import torch

from src.data import Batch
from src.models import MLP, imfDiT
from src.objectives import imf

VENDORED = Path(__file__).resolve().parents[1] / "src" / "models"
UPSTREAM_SHA256 = {
    "imfDiT.py": "9e607b251e58264403d61574fb99ed86dbe3e5fbc4eb21aa38f8dcad14c64309",
    "embedder.py": "e8975428281faea5cb05b807a408c65609628230ca34dfad01dd84d46fc5d676",
    "torch_models.py": "a4369cee16b9e812717170c844a9882b1c31fc1b5afe016b3aed06e74c290aa4",
}


@pytest.mark.parametrize("name", UPSTREAM_SHA256)
def test_vendored_code_is_upstream_up_to_import_paths(name):
    text = (VENDORED / name).read_text()
    text = text.replace("from .embedder import", "from models.embedder import")
    text = text.replace("from .torch_models import", "from models.torch_models import")
    assert hashlib.sha256(text.encode()).hexdigest() == UPSTREAM_SHA256[name]


def conditioning(n, num_classes):
    """(t, h, omega, t_min, t_max, y) for a batch of n."""
    t = torch.rand(n)
    y = torch.randint(0, num_classes + 1, (n,))
    return t, t * torch.rand(n), torch.full((n,), 2.0), torch.zeros(n), torch.ones(n), y


def test_mlp_heads_start_at_zero_like_imfdit():
    torch.manual_seed(0)
    net = MLP(dim=2, hidden=32, depth=3, aux_head_depth=1, num_classes=4)
    u, v = net(torch.randn(8, 2), *conditioning(8, 4))
    assert u.shape == v.shape == (8, 2) and (u == 0).all() and (v == 0).all()


def tiny_dit():
    torch.manual_seed(0)
    return imfDiT.imfDiT(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2,
                         num_heads=4, num_classes=10, aux_head_depth=1)


def test_imfdit_maps_images_to_two_heads():
    u, v = tiny_dit()(torch.randn(3, 4, 8, 8), *conditioning(3, 10))
    assert u.shape == v.shape == (3, 4, 8, 8)


def test_guided_imf_loss_runs_on_images():
    net = tiny_dit()
    batch = Batch(x=torch.randn(8, 4, 8, 8), sigma=torch.zeros(8), cond=torch.randint(0, 10, (8,)))
    loss, _ = imf.loss(net, batch, P_mean=-0.4, P_std=1.0, data_proportion=0.5, cfg_beta=1.0,
                       class_dropout_prob=0.1, norm_p=1.0, norm_eps=0.01, guidance=True)
    loss.backward()
    assert torch.isfinite(loss)
