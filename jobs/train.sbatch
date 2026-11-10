#!/usr/bin/env bash
# One GPU job running one command from the repo root, in env nrfm. Submit from
# the worktree root through the gate, passing the command in CMD:
#
#   research/bin/submit <slug> jobs/train.sbatch \
#     --export=ALL,CMD="python -m scripts.train --config configs/toy.yaml slug=<slug> seed=42"
#
# --export splits on commas and $CMD on spaces, so an override containing either
# belongs in a config file. The header below is the budget submit checks; copy this
# file into research/experiments/<slug>/ to change it.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail

# env nrfm has no activation hooks, so PATH alone activates it; loading the
# conda module instead costs minutes on network storage.
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo "$(date +%T) $(hostname) $(git rev-parse --short HEAD) $CMD"
$CMD
