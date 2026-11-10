#!/usr/bin/env bash
# Pilot: does iMF with its official hyperparameters (lr 1e-4, EMA 0.9999, p = 1)
# learn the clean checkerboard within the configs/toy.yaml budget? Seed 42.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
python -m scripts.train --config configs/toy.yaml slug=ambient-flowmap seed=42 runs_dir=research/runs
