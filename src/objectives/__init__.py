"""Training objectives, each `loss(net, batch, **kwargs) -> (scalar, logs)`.

A new objective is one module here plus one entry below.
"""

from src.objectives import ambient_imf, imf

OBJECTIVES = {"imf": imf.loss, "ambient_imf": ambient_imf.loss}
