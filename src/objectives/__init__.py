"""Training objectives, each `loss(net, batch, **kwargs) -> (scalar, logs)`.

A new objective is one new module here plus one entry below.
"""

from src.objectives import flow_matching, meanflow

OBJECTIVES = {
    "flow_matching": flow_matching.loss,
    "meanflow": meanflow.loss,
}
