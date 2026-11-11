"""Our iMF port against the official code itself, not against our reading of it.

- Objective: `src.objectives.imf.loss` reproduces the official JAX `iMeanFlow.forward`
  (golden outputs in tests/reference/imf_reference.npz, made by
  tests/reference/make_imf_reference.py) given the same network and the same random
  draws: loss, both loss terms and every parameter gradient.
- Sampler: `src.sampling.imf` reproduces the official PyTorch `iMeanFlow.generate`
  (tests/reference/upstream_imf_torch.py), run here.
- Networks: like imfDiT, the toy MLP does not condition on t, only on h = t - r.
"""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import Batch
from src.models import MLP, imfDiT
from src.objectives import imf
from src.sampling import imf as sample

REF_DIR = Path(__file__).resolve().parent / "reference"
UPSTREAM_SHA256 = {  # github.com/Lyy-iiis/imeanflow: imf.py on main (bf60cd7) and torch (0468798)
    "upstream_imf_jax.py": "8203e692a21f6e61e2b18d24579f85b3f04e6608a9fb419972e44718cebeb2f5",
    "upstream_imf_torch.py": "33cdfa1baa4f2725f0181f993ee07f5e15fb7c55fbe5a2f303db0b238b7f436b",
}
REF = np.load(REF_DIR / "imf_reference.npz")
CASES = sorted({k.split("/")[0] for k in REF.files if k.endswith("/loss")})
K = 3  # classes; label K is the null class


class ToyNet(torch.nn.Module):
    """The closed-form network of make_imf_reference.py, with its parameters."""

    num_classes = K

    def __init__(self):
        super().__init__()
        for k in REF.files:
            if k.startswith("param/"):
                self.register_parameter(k[6:], torch.nn.Parameter(torch.tensor(REF[k])))

    def forward(self, x, t, h, omega, t_min, t_max, y):
        f = torch.tanh(x.flatten(1) @ self.A + t[:, None] * self.b_t + h[:, None] * self.b_h
                       + (1 - 1 / omega)[:, None] * self.b_w + t_min[:, None] * self.b_s
                       + t_max[:, None] * self.b_e + self.E[y])
        return (f @ self.B_u).reshape(x.shape), (f @ self.B_v).reshape(x.shape)


@pytest.mark.parametrize("name", UPSTREAM_SHA256)
def test_reference_copies_are_upstream(name):
    assert hashlib.sha256((REF_DIR / name).read_bytes()).hexdigest() == UPSTREAM_SHA256[name]


def replay(monkeypatch, draws):
    """Serve torch.randn / randn_like / rand from the recorded draws, in order and by kind."""
    queue = list(draws)

    def take(kind, shape):
        got, value = queue.pop(0)
        assert got == kind, f"our code asked for a {kind} draw, upstream drew {got} here"
        return torch.tensor(value).reshape(shape)

    size = lambda args: args[0] if len(args) == 1 and not isinstance(args[0], int) else args
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: take("normal", size(a)))
    monkeypatch.setattr(torch, "randn_like", lambda x, **kw: take("normal", x.shape))
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: take("uniform", size(a)))
    return queue


@pytest.mark.parametrize("case", CASES)
def test_objective_matches_official_jax(monkeypatch, case):
    guided, norm_p = case.startswith("guided"), float(case.split("-p")[1])
    keys = sorted((k for k in REF.files if k.startswith(f"{case}/draw")),
                  key=lambda k: int(k.split("/")[1][4:]))
    draws = [(k.split("/")[2], REF[k]) for k in keys]
    if not guided:  # guidance=False draws only t, r and e; upstream's cfg draws fix omega = 1
        draws = draws[:3]
    queue = replay(monkeypatch, draws)

    net = ToyNet()
    x = torch.tensor(REF["x"])
    batch = Batch(x=x, sigma=torch.zeros(len(x)), cond=torch.tensor(REF[f"{case}/labels"]))
    loss, logs = imf.loss(net, batch, P_mean=-0.4, P_std=1.0, data_proportion=0.5, cfg_beta=1.0,
                          class_dropout_prob=0.1, norm_p=norm_p, norm_eps=0.01, guidance=guided)
    loss.backward()
    assert not queue, "upstream drew more random numbers than our code used"

    close = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    close(loss.item(), REF[f"{case}/loss"])
    close(logs["loss_u"].item(), REF[f"{case}/loss_u"])
    close(logs["loss_v"].item(), REF[f"{case}/loss_v"])
    for k, p in net.named_parameters():
        close(p.grad.numpy(), REF[f"{case}/grad/{k}"])


def upstream_torch(monkeypatch):
    stub = types.ModuleType("models")
    stub.imfDiT = types.ModuleType("models.imfDiT")
    monkeypatch.setitem(sys.modules, "models", stub)
    monkeypatch.setitem(sys.modules, "models.imfDiT", stub.imfDiT)
    spec = importlib.util.spec_from_file_location("upstream_imf_torch", REF_DIR / "upstream_imf_torch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_sampler_matches_official_torch_generate(monkeypatch, steps):
    up, net = upstream_torch(monkeypatch), ToyNet()
    official = up.iMeanFlow.__new__(up.iMeanFlow)
    torch.nn.Module.__init__(official)
    official.net, official.dtype, official.img_channels, official.img_size = net, torch.float32, 1, 2

    eps = torch.tensor(REF["x"]).reshape(-1, 1, 2, 2)
    y = torch.tensor(REF["guided-p1/labels"])
    rng = types.SimpleNamespace(randn=lambda shape: eps.clone())
    with torch.no_grad():
        expected = official.generate(len(eps), rng, steps, omega=2.0, t_min=0.2, t_max=0.8, labels=y)
    assert torch.equal(sample(net, eps, steps, y, omega=2.0, t_min=0.2, t_max=0.8), expected)


@pytest.mark.parametrize("make", [
    lambda: MLP(dim=2, hidden=32, depth=3, aux_head_depth=1, num_classes=K),
    lambda: imfDiT.imfDiT(input_size=8, patch_size=2, in_channels=1, hidden_size=64, depth=2,
                          num_heads=4, num_classes=K, aux_head_depth=1),
])
def test_networks_ignore_t_like_imfdit(make):
    torch.manual_seed(0)
    net = make()
    for p in net.parameters():  # zero-initialised heads would make any input look ignored
        torch.nn.init.normal_(p, std=0.05)
    n = 4
    z = torch.randn(n, 2) if isinstance(net, MLP) else torch.randn(n, 1, 8, 8)
    rest = (torch.full((n,), 0.3), torch.full((n,), 2.0), torch.zeros(n), torch.ones(n),
            torch.arange(n) % (K + 1))
    with torch.no_grad():
        a, b = net(z, torch.full((n,), 0.1), *rest), net(z, torch.full((n,), 0.9), *rest)
        c = net(z, torch.full((n,), 0.1), torch.full((n,), 0.6), *rest[1:])
    assert all(torch.equal(p, q) for p, q in zip(a, b))       # t has no effect
    assert not torch.allclose(a[0], c[0])                     # h does


def test_mlp_condition_on_t_makes_t_an_input():
    torch.manual_seed(0)
    net = MLP(dim=2, hidden=32, depth=3, aux_head_depth=1, num_classes=K, condition_on_t=True)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    n, z = 4, torch.randn(4, 2)
    rest = (torch.full((n,), 0.3), torch.full((n,), 2.0), torch.zeros(n), torch.ones(n), torch.zeros(n, dtype=int))
    with torch.no_grad():
        assert not torch.allclose(net(z, torch.full((n,), 0.1), *rest)[0], net(z, torch.full((n,), 0.9), *rest)[0])
