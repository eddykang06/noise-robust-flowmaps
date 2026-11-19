#!/usr/bin/env bash
# Draws this direction's figures (figures.py) into $NRFM_OUT/ambient-flowmap/figures/.
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export MPLCONFIGDIR=$TMPDIR
cd "$SLURM_SUBMIT_DIR"
echo "$(git rev-parse --short HEAD)"
python research/experiments/ambient-flowmap/figures.py /n/scratch/users/e/edk005/nrfm/ambient-flowmap/figures
