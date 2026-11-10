#!/usr/bin/env bash
# Runs the test suite inside the research-free main candidate checkout, to show
# the code runs with no research/ orchestration present.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/scratch/users/e/edk005/nrfm/main-candidate
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
git log -1 --oneline
ls research CLAUDE.md .claude 2>&1 | head -3
python -m pytest -v -rs
