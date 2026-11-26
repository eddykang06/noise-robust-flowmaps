#!/usr/bin/env bash
# Short end-to-end check of main's iMF ImageNet setup on the real assembled posteriors: 50 steps
# of a 2-block imfDiT on 2,048 training images with Gaussian noise (fixed-noise cache plus
# per-step posterior draws), scored by FID/precision/recall/IS on 1,000 class-balanced validation
# latents (the fewest a class-balanced reference allows) at 1 sampling step, with the
# true-vs-generated figure. Bare bones by design. Memory: 1.5x GPU smoke 60707401 (1.54 GB).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data NRFM_OUT=/n/scratch/users/e/edk005/nrfm
OUT=$NRFM_OUT/imagenet-run-check/$SLURM_JOB_ID && mkdir -p $OUT
cd /n/scratch/users/e/edk005/nrfm/main-parquet && git log -1 --oneline && git status --short && nvidia-smi --query-gpu=name --format=csv,noheader
python -m scripts.train --config configs/imagenet.yaml slug=check out_root=$OUT runs_dir=$OUT \
  data.n=2048 "corruption={name: gaussian, sigma: 0.5, clean_frac: 0.5}" \
  model.depth=2 model.aux_head_depth=1 train.batch_size=32 train.steps=50 train.eval_every=50 \
  metric.n_samples=1000 sampler.batch=250 "sampler.steps=[1]"
tail -n 1 $OUT/check.jsonl | python -c "import json, sys; print(json.loads(sys.stdin.read())['metrics'])"
find $OUT -name true-vs-generated.png
