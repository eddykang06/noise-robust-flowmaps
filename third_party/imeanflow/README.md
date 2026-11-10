# Vendored: official improved MeanFlow (iMF) network

Source: https://github.com/Lyy-iiis/imeanflow, branch `torch`, commit
`04687983e821b3ad01f54f03dc44194a33c20c54` (2026-02-27), MIT license (`LICENSE`).
Paper: Geng, Lu, Wu, Shechtman, Kolter, He, "Improved Mean Flows: On the
Challenges of Fastforward Generative Models", arXiv:2512.02012.

`models/imfDiT.py`, `models/embedder.py` and `models/torch_models.py` are
copied unchanged except for their import lines, made package-relative:

    imfDiT.py:   from models.embedder ...     -> from .embedder ...
                 from models.torch_models ... -> from .torch_models ...
    embedder.py: from models.torch_models ... -> from .torch_models ...

The `torch` branch is inference-only. The training objective is not vendored:
`src/objectives/imf.py` ports it from the official JAX `imf.py` (branch `main`,
commit `bf60cd7cb653f6628e59d48034b333c5eba445e2`), whose defaults it keeps.
Do not edit these files; update them by re-copying from upstream.

Upstream SHA-256 (verified by `tests/test_models.py`):

    9e607b251e58264403d61574fb99ed86dbe3e5fbc4eb21aa38f8dcad14c64309  models/imfDiT.py
    e8975428281faea5cb05b807a408c65609628230ca34dfad01dd84d46fc5d676  models/embedder.py
    a4369cee16b9e812717170c844a9882b1c31fc1b5afe016b3aed06e74c290aa4  models/torch_models.py
