#!/usr/bin/env bash
# Brief's first experiment: MeanFlow on clean, corrupted (sigma 0.3, 10% clean,
# noisy samples used as clean) and clean-only data; seeds 42-44, 20k steps each.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

D=research/experiments/ambient-flowmap
for seed in 42 43 44; do
  python -m scripts.train --config configs/toy.yaml slug=ambient-flowmap seed=$seed
  python -m scripts.train --config $D/corrupt.yaml slug=ambient-flowmap seed=$seed
  python -m scripts.train --config $D/corrupt.yaml slug=ambient-flowmap seed=$seed data.keep_noisy=false
done
