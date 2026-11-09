# Noise-Robust Flow Maps

Official Implementation of "Noise-Robust Flow Maps." In progress.

## Layout

```
configs/       one YAML per dataset; every budget explicit
src/
  config.py    Config dataclass; YAML + dotted key=value overrides
  data.py      Batch (x [B, ...], sigma, mask, cond) and datasets
  toy.py       2-D toy distributions
  corrupt.py   corruption models -> per-sample sigma
  models.py    flow-map networks net(z, r, t, cond) -> u
  path.py      the path z_t = (1 - t) x + t eps
  objectives/  one module per training objective
  sampling.py  samplers (flow map, Euler)
  metrics.py   sample-quality metrics
  training.py  the shared training loop
scripts/       train, eval, sample, check: thin CLIs
jobs/          Slurm template
```

Each component section of a config names its implementation (`name:`) and
passes the remaining keys as keyword arguments. A new objective is one module
in `src/objectives/` plus one entry in `OBJECTIVES`.

## Usage

Run from the repo root, on a GPU node, in conda env `nrfm`:

```sh
python -m scripts.train --config configs/toy.yaml slug=<slug> seed=42 objective.ratio=0.5
python -m scripts.eval  <run_dir> --steps 1 2 4
python -m scripts.sample <run_dir> --n 10000 --steps 1 --seed 0
python -m scripts.check          # correctness and reproducibility checks
```

On O2, submit through the gate (see `jobs/train.sbatch`):

```sh
research/bin/submit <slug> jobs/train.sbatch \
  --export=ALL,CMD="python -m scripts.train --config configs/toy.yaml slug=<slug> seed=42"
```

A run lives in `/n/scratch/users/e/edk005/nrfm/<slug>/<run_id>/`. The
`run_id` is a hash of the resolved config, so resubmitting the same command
resumes from the last checkpoint. Each evaluation milestone appends one record
(git SHA, job, seed, step, wall-clock, metrics, config) to
`research/runs/<slug>.jsonl`. Training is deterministic: the same config and
seed reproduce the same metrics bitwise on the same hardware.
