# Noise-Robust Flow Maps

Official Implementation of "Noise-Robust Flow Maps." In progress.

The flow map is improved MeanFlow (iMF; Geng et al., arXiv:2512.02012). This
repo wraps the official code: its PyTorch network is vendored unchanged in
`third_party/imeanflow/`, and `src/objectives/imf.py` ports its JAX training
objective, keeping the official defaults.

## Layout

```
configs/       one YAML per dataset; every budget explicit
src/
  config.py    Config dataclass; YAML + dotted key=value overrides
  data.py      Batch (x [B, ...], sigma, mask, cond) and datasets
  toy.py       2-D toy distributions
  corrupt.py   corruption models -> per-sample sigma
  models.py    networks net(z, t, h, omega, t_min, t_max, y) -> (u, v)
  path.py      the path z_t = (1 - t) x + t e
  objectives/  training objectives (imf)
  sampling.py  iMF sampler
  metrics.py   sample-quality metrics
  training.py  the shared training loop
third_party/   vendored official iMF network (do not edit)
scripts/       train, eval, sample: thin CLIs
tests/         pytest suite; GPU tests are marked `gpu`
jobs/          Slurm template
logs/          Slurm job logs (untracked)
```

Each component section of a config names its implementation (`name:`) and
passes the remaining keys as keyword arguments. A new objective is one module
in `src/objectives/` plus one entry in `OBJECTIVES`.

## Setup

```sh
conda create -n nrfm python=3.11 && conda activate nrfm
pip install -r requirements.txt
```

## Usage

Run from the repo root, on a GPU node, in conda env `nrfm`, with `NRFM_OUT`
set to where run directories should go:

```sh
export NRFM_OUT=/n/scratch/users/${USER:0:1}/$USER/nrfm
python -m scripts.train --config configs/toy.yaml slug=<name> seed=42 objective.data_proportion=0.75
python -m scripts.eval  <run_dir> --steps 1 2 4
python -m scripts.sample <run_dir> --n 10000 --steps 1 --seed 0
python -m pytest                 # tests; GPU tests are skipped without CUDA
```

On a Slurm cluster (see `jobs/train.sbatch`):

```sh
sbatch --export=ALL,CMD="python -m scripts.train --config configs/toy.yaml slug=<name> seed=42" \
  jobs/train.sbatch
```

The job template sets `NRFM_OUT` to the submitting user's scratch space and
writes Slurm logs to `logs/`. A run lives in `$NRFM_OUT/<slug>/<run_id>/`. The
`run_id` is a hash of the config without its paths, so resubmitting the same
command resumes from the last checkpoint. Each evaluation milestone appends one
record (git SHA, job, seed, step, wall-clock, metrics, config) to
`<runs_dir>/<slug>.jsonl`, `runs/` (untracked) by default. Paths in a config may
use environment variables and `~`; relative ones are taken from the repo root.
Training is deterministic: the same config and seed reproduce the same metrics
bitwise on the same hardware.
