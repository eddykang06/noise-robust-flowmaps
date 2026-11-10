#!/usr/bin/env bash
# Bitwise behaviour check of a refactor: the same seeded fingerprint (models,
# objectives, sampler) in the pinned pre-change tree and in this tree, then the
# non-GPU tests on this tree.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NEW="$SLURM_SUBMIT_DIR" OLD=/n/scratch/users/e/edk005/nrfm/equiv-before
FP=$NEW/research/experiments/ambient-flowmap/fingerprint.py
(cd $OLD && git log -1 --oneline && python $FP /tmp/fp-old-$SLURM_JOB_ID.json > /dev/null)
(cd $NEW && git log -1 --oneline && python $FP /tmp/fp-new-$SLURM_JOB_ID.json > /dev/null)
diff /tmp/fp-old-$SLURM_JOB_ID.json /tmp/fp-new-$SLURM_JOB_ID.json && echo "FINGERPRINTS IDENTICAL"
cd $NEW && python -m pytest -v -rs -m "not gpu"
