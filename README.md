# Noise-Robust Flow Maps

Official Implementation of "Noise-Robust Flow Maps."

Clean training data is scarce in science. Noise-robust flow
maps learn to generate clean samples in one or a few steps, trained directly on noisy
or corrupted data. Built on improved MeanFlow (iMF; Geng et al., 2025) and ambient
diffusion's exact training on Gaussian-noisy data (Daras et al., 2024, "Consistent Diffusion
Meets Tweedie").

## Setup

Requires Python 3.11 and an NVIDIA GPU. The pinned PyTorch build (CUDA 12.6) supports
V100 and newer.

```sh
git clone https://github.com/eddykang06/noise-robust-flowmaps.git
cd noise-robust-flowmaps
conda create -n nrfm python=3.11 && conda activate nrfm
pip install -r requirements.txt
export NRFM_OUT=/path/to/outputs        # where runs are saved
export NRFM_DATA=/path/to/data          # prepared datasets and metric networks (default: data/)
```

**Data.** The spiral is generated on the fly. Prepare the other datasets once:

```sh
python -m scripts.prepare_data mnist                        # digits + the MNIST metric classifier
python -m scripts.prepare_data inception                    # FID Inception network
python -m scripts.prepare_data imagenet-download           # SD-VAE posteriors, 41 GB
python -m scripts.prepare_data imagenet-assemble --split train
python -m scripts.prepare_data imagenet-assemble --split val
```

ImageNet is ILSVRC-2012 at 256 x 256 in the SD-VAE ft-MSE latent space, set up as iMF does: each
image's VAE posterior (mean and std, precomputed with iMF's preparation:
`ChocolateDave/imagenet-256-sd-vae-latents`) is normalized with iMF's per-channel constants, and
training draws a fresh latent from it every step. The cache is derived from ImageNet, whose terms
of use (non-commercial research) apply. Noise is added to the normalized latents.

## Usage

Run all commands from the repository root.

**Configure.** Each experiment is a YAML file in `configs/` that sets the data,
corruption, model, objective, sampler, metric and training budget. Any value can be
overridden on the command line as a dotted `key=value`, for example `train.steps=20000`.
`configs/imf.yaml` trains iMF and `configs/ambient_imf.yaml` ambient iMF on a two-armed
spiral observed in 64 dimensions (the Pixel MeanFlow toy: the data lie on a fixed random
plane, `src/embed.py`); `configs/mnist.yaml` and `configs/imagenet.yaml` train
class-conditional iMF on MNIST and on ImageNet 256 latents. Noise is added with a corruption
override, for example `corruption="{name: gaussian, sigma: 1.0, clean_frac: 0.75}"`
(Gaussian or uniform, `src/corrupt.py`); each sample keeps one fixed noise draw. Standard iMF
treats noisy samples as clean; ambient iMF trains on each noisy sample only at times above its
noise level, with a target whose expectation is the clean one
(`src/objectives/ambient_imf.py`). Spiral runs
report W2 in all dimensions (`wasserstein2_full`), in the plane (`wasserstein2`) and the RMS
off the plane (`off_plane_rms`); image runs report FID, precision, recall and the Inception
score (`src/metrics.py`). The image configs fit one GPU: MNIST (batch 64) needs about 17 GB,
ImageNet (batch 32) about 30 GB, so a 40 GB or larger GPU.

**Class-conditional.** `objective.guidance: true` trains with classifier-free guidance as iMF
does (a random guidance scale and interval per sample, 10% of labels dropped); the MNIST and
ImageNet configs set it. On the spiral the two arms are the classes:
`model.num_classes=2 objective.guidance=true`, scored also by `arm_accuracy`. Ambient iMF
guides its ambient target the same way. Conditional runs sample the reference's labels
(class-balanced for ImageNet). `sampler.omega` may list guidance scales, for example
`sampler.omega="[1.0, 2.0, 3.0]"`; each is scored under `<metric>@<steps>_w<omega>`.

**Train.** Trains a model and evaluates it at fixed intervals. `slug` names the experiment.

```sh
python -m scripts.train --config configs/ambient_imf.yaml slug=demo seed=42 \
  corruption="{name: gaussian, sigma: 1.0, clean_frac: 0.75}"
```

The run directory, `$NRFM_OUT/<slug>/<run_id>/`, is printed at the start and holds the
config, checkpoint and a figure of true against generated samples. Metrics are appended to `runs/<slug>.jsonl`. Rerunning the same
command resumes from the last checkpoint, and the same config and seed reproduce
results exactly on the same hardware.

**Evaluate.** Rescores a run's checkpoint, optionally at other numbers of sampling steps.

```sh
python -m scripts.eval <run_dir> --steps 1 2
```

**Sample.** Draws samples from a run's checkpoint at guidance scale `--omega` and saves them,
with their labels (balanced over the classes for conditional runs), to the run directory as
`samples-<steps>step.pt`.

```sh
python -m scripts.sample <run_dir> --n 10000 --steps 1 --omega 1.0 --seed 0
```

**Plot.** Draws true against generated samples for one or more runs.

```sh
python -m scripts.plot <run_dir> [<run_dir> ...] --out runs.png
```

**Test.** `python -m pytest`; the GPU tests are skipped without CUDA.

**On Slurm.** `scripts/train.sbatch` runs any command above as a single-GPU job in the
`nrfm` env and writes job logs to `logs/`. Set `NRFM_OUT` before submitting (it defaults
to O2 scratch space).

```sh
sbatch --export=ALL,CMD="python -m scripts.train --config configs/imf.yaml slug=demo seed=42" scripts/train.sbatch
```

## Layout

```
configs/   iMF and ambient iMF on the spiral; iMF on MNIST and ImageNet
src/       datasets, corruption, models, objectives, sampling, metrics, training loop
scripts/   train, evaluate, sample, plot, prepare data, and the Slurm job template
tests/     pytest suite; tests/reference holds the upstream iMF reference outputs
```
