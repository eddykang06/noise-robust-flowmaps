"""Flow-map networks with the iMF signature

    net(z, t, h, omega, t_min, t_max, y) -> (u, v)

h = t - r, u is the average velocity over [r, t], v the auxiliary velocity head,
omega and [t_min, t_max] the guidance scale and interval, y the class label
(`num_classes` is the null class). As in iMF, networks see h but not t.

`imfDiT.py`, `embedder.py` and `torch_models.py` are the official iMF network
(github.com/Lyy-iiis/imeanflow, branch `torch`, commit 0468798, MIT license in
`LICENSE-imeanflow`), copied unchanged except for package-relative imports
(`from models.x` -> `from .x`); `tests/test_models.py` checks them against the
upstream SHA-256. `mlp.py` is this repo's toy analogue.
"""

from src.models import imfDiT
from src.models.mlp import MLP

MODELS = {
    "mlp": MLP,
    "imfDiT_B_2": imfDiT.imfDiT_B_2,
    "imfDiT_M_2": imfDiT.imfDiT_M_2,
    "imfDiT_L_2": imfDiT.imfDiT_L_2,
    "imfDiT_XL_2": imfDiT.imfDiT_XL_2,
}
