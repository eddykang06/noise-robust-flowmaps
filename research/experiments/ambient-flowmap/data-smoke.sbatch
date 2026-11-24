#!/usr/bin/env bash
# Smoke test of main's real-data pipeline (tree NEW): prepares MNIST (download, metric
# classifier) and the FID Inception network into the lasting data root, trains MNIST iMF for
# 2,000 steps with FID evaluation, then prepares a fake ImageNet (random JPEGs in class
# folders) through the SD-VAE and trains on it briefly with Inception metrics; finally the
# full test suite. Memory is an estimate (no comparable job measured yet).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data
NEW=/n/scratch/users/e/edk005/nrfm/main-consolidate
OUT=$NRFM_OUT/data-smoke/$SLURM_JOB_ID && mkdir -p $OUT
cd $NEW && nvidia-smi --query-gpu=name --format=csv,noheader && git log -1 --oneline
RUN="slug=data-smoke out_root=$OUT/runs runs_dir=$OUT"

python -m scripts.prepare_data mnist
python -m scripts.prepare_data inception
python -m scripts.train --config configs/mnist.yaml $RUN train.steps=2000 train.eval_every=1000 metric.n_samples=2000

python - $OUT/fake <<'PY'
import sys, numpy as np
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(0)
for split, n in [("train", 20), ("val", 8)]:
    for c in ["n01", "n02", "n03"]:
        d = Path(sys.argv[1]) / split / c
        d.mkdir(parents=True)
        for i in range(n):
            h, w = rng.integers(200, 600, 2)
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(d / f"{i}.JPEG")
PY
mkdir -p $OUT/root && mkdir -p $NRFM_DATA/weights && ln -s $NRFM_DATA/weights $OUT/root/weights
for split in train val; do NRFM_DATA=$OUT/root python -m scripts.prepare_data imagenet --src $OUT/fake/$split --split $split; done
NRFM_DATA=$OUT/root python -m scripts.train --config configs/imagenet.yaml $RUN train.steps=50 train.eval_every=50 \
  metric.n_samples=24 sampler.batch=12 model.depth=2 model.aux_head_depth=1 \
  "corruption={name: gaussian, sigma: 0.5, clean_frac: 0.5}"
tail -n 2 $OUT/data-smoke.jsonl | cut -c1-600
python -m pytest -rs
