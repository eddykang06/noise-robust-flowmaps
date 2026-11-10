#!/usr/bin/env bash
# CPU-only tests: the ambient objective on this branch, and every non-GPU test
# in the research-free main candidate. GPU tests run separately (job 59171059).
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
status=0
git log -1 --oneline
python -m pytest -v -rs tests/test_ambient_imf.py tests/test_imf.py || status=1
cd /n/scratch/users/e/edk005/nrfm/main-candidate
git log -1 --oneline
python -m pytest -v -rs -m "not gpu" || status=1
exit $status
