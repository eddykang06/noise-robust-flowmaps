"""Experiment config: one YAML file plus dotted `key=value` overrides.

Each component section (data, corruption, model, objective, sampler, metric)
holds a `name`, which selects the implementation, and the keyword arguments
passed to it.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    slug: str
    seed: int
    data: dict
    corruption: dict
    model: dict
    objective: dict
    sampler: dict
    metric: dict
    train: dict
    out_root: str

    @property
    def run_id(self) -> str:
        """Deterministic, so rerunning the same config resumes the same run."""
        digest = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return f"{self.objective['name']}-s{self.seed}-{digest[:8]}"


def load(path: str, overrides: list[str] = ()) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = raw
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return Config(**raw)


def kwargs(section: dict) -> dict:
    """A component section without its `name`."""
    return {k: v for k, v in section.items() if k != "name"}
