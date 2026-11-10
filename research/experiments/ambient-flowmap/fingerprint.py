"""Seeded outputs of the models, objectives and sampler, for checking that a
refactor leaves behaviour bitwise unchanged. Run from a repo root:
    python <this file> <out.json>
"""

import hashlib
import json
import sys

import torch

from src.data import Batch
from src.models import MODELS
from src.objectives import OBJECTIVES
from src.sampling import SAMPLERS

IMF = dict(P_mean=-0.4, P_std=1.0, data_proportion=0.5, norm_p=1.0, norm_eps=0.01)


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def net(name, **kw):
    torch.manual_seed(0)
    return MODELS[name](**kw)


out = {}
mlp = net("mlp", dim=2, hidden=64, depth=4, aux_head_depth=1, num_classes=4)
dit = net("imfDiT_B_2", input_size=8, in_channels=4, num_classes=10, depth=2, aux_head_depth=1)
for name, m in [("mlp", mlp), ("dit", dit)]:
    out[f"{name}_params"] = digest(*m.state_dict().values())
    for p in m.parameters():  # make every head non-zero
        torch.nn.init.normal_(p, std=0.3, generator=torch.Generator().manual_seed(1))

torch.manual_seed(2)
toy = Batch(x=torch.randn(256, 2), sigma=torch.zeros(256), cond=torch.randint(0, 4, (256,)))
img = Batch(x=torch.randn(4, 4, 8, 8), sigma=torch.zeros(4), cond=torch.randint(0, 10, (4,)))
noisy = Batch(x=torch.randn(256, 2), sigma=torch.full((256,), 0.3))
cases = [
    ("imf_uncond", mlp, toy, "imf", dict(cfg_beta=1.0, class_dropout_prob=0.1, guidance=False)),
    ("imf_guided", mlp, toy, "imf", dict(cfg_beta=1.0, class_dropout_prob=0.1, guidance=True)),
    ("imf_dit", dit, img, "imf", dict(cfg_beta=1.0, class_dropout_prob=0.1, guidance=True)),
    ("ambient", mlp, noisy, "ambient_imf", dict(t_margin=0.05)),
]
for name, m, batch, objective, kw in cases:
    m.zero_grad()
    torch.manual_seed(3)
    loss, _ = OBJECTIVES[objective](m, batch, **IMF, **kw)
    loss.backward()
    out[name] = [loss.item(), digest(*(p.grad for p in m.parameters() if p.grad is not None))]

eps = torch.randn(512, 2, generator=torch.Generator().manual_seed(4))
out["sample"] = digest(SAMPLERS["imf"](mlp, eps, 2, None, omega=1.0, t_min=0.0, t_max=1.0))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
