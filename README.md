# Noise-Robust Flow Maps

Official Implementation of "Noise-Robust Flow Maps."

Clean training data is scarce in science. Noise-robust flow
maps learn to generate clean samples in one or a few steps, trained directly on noisy
or corrupted data. Built on improved MeanFlow (Geng et al., 2025) and ambient
diffusion (Daras et al., 2023).

## Setup

Requires Python 3.11 and an NVIDIA GPU. The pinned PyTorch build (CUDA 12.6) supports
V100 and newer.

```sh
git clone https://github.com/eddykang06/noise-robust-flowmaps.git
cd noise-robust-flowmaps
conda create -n nrfm python=3.11 && conda activate nrfm
pip install -r requirements.txt
export NRFM_OUT=/path/to/outputs        # where runs are saved
```

## Usage

Run all commands from the repository root.

**Configure.** Each experiment is a YAML file in `configs/` that sets the data,
corruption, model, objective, sampler, metric and training budget. Any value can be
overridden on the command line as a dotted `key=value`, for example `train.steps=20000`.

**Train.** Trains a model and evaluates it at fixed intervals. `slug` names the experiment.

```sh
python -m scripts.train --config configs/toy.yaml slug=toy seed=42
```

The run directory, `$NRFM_OUT/<slug>/<run_id>/`, is printed at the start and holds the
config and checkpoint. Metrics are appended to `runs/<slug>.jsonl`. Rerunning the same
command resumes from the last checkpoint, and the same config and seed reproduce
results exactly on the same hardware.

**Evaluate.** Rescores a run's checkpoint, optionally at other numbers of sampling steps.

```sh
python -m scripts.eval <run_dir> --steps 1 2
```

**Sample.** Draws samples from a run's checkpoint and saves them to the run directory
as `samples-<steps>step.pt`.

```sh
python -m scripts.sample <run_dir> --n 10000 --steps 1 --seed 0
```

**On Slurm.** `jobs/train.sbatch` runs any command above as a single-GPU job in the
`nrfm` env and writes job logs to `logs/`. Set `NRFM_OUT` before submitting (it defaults
to O2 scratch space).

```sh
sbatch --export=ALL,CMD="python -m scripts.train --config configs/toy.yaml slug=toy seed=42" jobs/train.sbatch
```

## Layout

```
configs/   experiment configs
src/       data, corruption, models, objectives, sampling, metrics, training loop
scripts/   train, evaluate, sample
jobs/      Slurm job template
```
