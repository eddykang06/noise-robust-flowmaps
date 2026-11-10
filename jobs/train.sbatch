#!/usr/bin/env bash
# One GPU job running one command from the repo root, in env nrfm. Submit from
# the repo root, passing the command in CMD:
#
#   sbatch --export=ALL,CMD="python -m scripts.train --config configs/toy.yaml slug=<name> seed=42" \
#     jobs/train.sbatch
#
# --export splits on commas and $CMD on spaces, so an override containing either
# belongs in a config file. Copy this file to change the resources below; toy runs
# peak at ~0.8G of RAM (sacct -o MaxRSS).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

# env nrfm has no activation hooks, so PATH alone activates it; loading the
# conda module instead costs minutes on network storage.
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
# Run directories default to the submitting user's O2 scratch space.
export NRFM_OUT="${NRFM_OUT:-/n/scratch/users/${USER:0:1}/$USER/nrfm}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo "$(date +%T) $(hostname) $(git rev-parse --short HEAD) $CMD"
$CMD
