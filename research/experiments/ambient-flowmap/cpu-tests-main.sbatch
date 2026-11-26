#!/usr/bin/env bash
# Non-GPU tests of a main checkout (TREE, default main's worktree). Sized from CPU check
# 60841856 (1 min, under 1 GB).
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ${TREE:-/n/scratch/users/e/edk005/nrfm/main-consolidate}
git log -1 --oneline && git status --short
python -m pytest -q -rs -m "not gpu"
