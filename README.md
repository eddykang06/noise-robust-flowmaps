# Noise-Robust Flow Maps

Official implementation of "Noise-Robust Flow Maps": one- or few-step generators of clean
samples, trained on partly noisy data. Built on improved MeanFlow (iMF; Geng et al., 2025) and
exact ambient training on noisy data (Daras et al., 2024, "Consistent Diffusion Meets Tweedie").

## Setup

Python 3.11 and an NVIDIA GPU (V100 or newer).

```sh
git clone https://github.com/eddykang06/noise-robust-flowmaps.git && cd noise-robust-flowmaps
conda create -n nrfm python=3.11 && conda activate nrfm
pip install -r requirements.txt
export NRFM_OUT=/path/to/runs     # run outputs
export NRFM_DATA=/path/to/data    # datasets and metric networks (default: data/)
```

Prepare datasets once (the spiral needs nothing):

```sh
python -m scripts.prepare_data mnist                            # MNIST + its metric classifier
python -m scripts.prepare_data inception                        # Inception network for FID
python -m scripts.prepare_data imagenet-download                # ImageNet VAE latents, 41 GB
python -m scripts.prepare_data imagenet-assemble --split train  # then --split val
```

## Configs

| config | data | model | GPU memory |
|---|---|---|---|
| `configs/imf.yaml` | two-armed spiral in 64-D | iMF (MLP) | small |
| `configs/ambient_imf.yaml` | same | ambient iMF (MLP) | small |
| `configs/mnist.yaml` | MNIST, class-conditional | iMF (DiT-B/4, batch 64) | ~17 GB |
| `configs/imagenet.yaml` | ImageNet 256 latents, class-conditional | iMF (DiT-B/2, batch 32) | ~30 GB |

- **Ambient iMF** trains on each noisy sample only at times above its noise level, with a target
  whose expectation is the clean one. Standard iMF treats noisy samples as clean.
- **ImageNet** follows iMF exactly: SD-VAE ft-MSE posteriors (precomputed:
  `ChocolateDave/imagenet-256-sd-vae-latents`), iMF's per-channel normalization, a fresh latent
  each step, no flips. ImageNet's terms of use apply.
- **Overrides:** any value as a dotted `key=value`, e.g. `train.steps=20000`.

| to | override |
|---|---|
| add noise (Gaussian or uniform, fixed per sample) | `corruption="{name: gaussian, sigma: 1.0, clean_frac: 0.75}"` |
| condition the spiral on its arm | `model.num_classes=2 objective.guidance=true` |
| score several guidance scales | `sampler.omega="[1.0, 2.0, 3.0]"` (keys `<metric>@<steps>_w<omega>`) |

**Metrics.** Spiral: W2 in 64-D (`wasserstein2_full`), in the plane (`wasserstein2`), and distance
off the plane (`off_plane_rms`). Images: FID, precision, recall, Inception score. Conditional runs
also report label accuracy (`arm_accuracy` on the spiral, `accuracy` on MNIST).

## Usage

Run from the repository root.

```sh
python -m scripts.train --config configs/ambient_imf.yaml slug=demo seed=42 \
  corruption="{name: gaussian, sigma: 1.0, clean_frac: 0.75}"     # train + evaluate
python -m scripts.eval <run_dir> --steps 1 2                       # rescore a checkpoint
python -m scripts.sample <run_dir> --n 10000 --steps 1 --omega 1.0 --seed 0
python -m scripts.plot <run_dir> [<run_dir> ...] --out runs.png
python -m pytest                                                   # tests; GPU tests skip without CUDA
```

- Runs live in `$NRFM_OUT/<slug>/<run_id>/` (config, checkpoint, figure); metrics append to
  `runs/<slug>.jsonl`. Rerunning a command resumes it; the same config and seed reproduce exactly
  on the same hardware.
- Samples are saved as `samples-<steps>step.pt`, with labels (class-balanced for conditional runs).
- On Slurm, `scripts/train.sbatch` runs any command as one GPU job (logs in `logs/`):

  ```sh
  sbatch --export=ALL,CMD="python -m scripts.train --config configs/imf.yaml slug=demo seed=42" scripts/train.sbatch
  ```

## Layout

```
configs/   experiment configs (above)
src/       datasets, corruption, models, objectives, sampling, metrics, training loop
scripts/   train, eval, sample, plot, prepare_data, Slurm template
tests/     pytest suite; tests/reference holds upstream iMF reference outputs
```
