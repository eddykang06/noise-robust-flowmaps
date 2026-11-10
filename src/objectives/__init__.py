"""Training objectives, each `loss(net, batch, **kwargs) -> (scalar, logs)`.

A new objective is one module here plus one entry below.
"""

from src.objectives import imf

OBJECTIVES = {"imf": imf.loss}
