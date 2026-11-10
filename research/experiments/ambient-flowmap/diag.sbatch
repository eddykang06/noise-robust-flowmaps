#!/usr/bin/env bash
# Diagnose MeanFlow degrading with training on the clean checkerboard (jobs
# 59156303-4): a flow-matching control, then MeanFlow with lower learning rate
# and/or without adaptive loss weighting. Seed 42, 20k steps each.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

common="slug=ambient-flowmap seed=42 train.eval_every=2500"
python -m scripts.train --config research/experiments/ambient-flowmap/fm.yaml $common
python -m scripts.train --config configs/toy.yaml $common train.lr=1.0e-4
python -m scripts.train --config configs/toy.yaml $common objective.p=0.0
python -m scripts.train --config configs/toy.yaml $common train.lr=1.0e-4 objective.p=0.0
