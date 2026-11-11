#!/usr/bin/env bash
# First experiment (brief, eddy 90d5ebd): five arms in parallel on one GPU, each
# running seeds 42-44 in sequence.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
source research/experiments/ambient-flowmap/arms.sh
LOGS=$NRFM_OUT/ambient-flowmap/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"

run_arm() {
  arm_args $1
  for seed in 42 43 44; do python -m scripts.train "${ARGS[@]}" seed=$seed; done
}

nvidia-smi --query-gpu=name --format=csv,noheader
echo "$(git rev-parse --short HEAD); logs in $LOGS"
all_arms run_arm
