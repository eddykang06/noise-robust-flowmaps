#!/usr/bin/env bash
# Non-GPU tests inside the research-free main candidate checkout.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/scratch/users/e/edk005/nrfm/main-candidate
git log -1 --oneline
python -m pytest -v -rs -m "not gpu"
