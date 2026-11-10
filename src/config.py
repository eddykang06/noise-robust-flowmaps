"""Experiment config: one YAML file plus dotted `key=value` overrides.

Each component section (data, corruption, model, objective, sampler, metric)
holds a `name`, which selects the implementation, and the keyword arguments
passed to it. The path fields `out_root` and `runs_dir` may use environment
variables and `~`; relative paths are taken from the repository root.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import yaml

PATHS = ("out_root", "runs_dir")


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
    out_root: str   # run directories: <out_root>/<slug>/<run_id>/
    runs_dir: str   # evaluation records: <runs_dir>/<slug>.jsonl

    @property
    def run_id(self) -> str:
        """Deterministic in everything but the paths, so rerunning the same
        experiment resumes the same run wherever its outputs live."""
        content = {k: v for k, v in asdict(self).items() if k not in PATHS}
        digest = hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
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
    for key in PATHS:
        raw[key] = expand(raw[key])
    return Config(**raw)


def expand(path: str) -> str:
    out = os.path.expanduser(os.path.expandvars(path))
    if "$" in out:
        raise ValueError(f"unset environment variable in path {path!r}")
    return out


def kwargs(section: dict) -> dict:
    """A component section without its `name`."""
    return {k: v for k, v in section.items() if k != "name"}
