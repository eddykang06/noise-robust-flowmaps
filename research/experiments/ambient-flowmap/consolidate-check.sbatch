#!/usr/bin/env bash
# Checks that the consolidated main tree (NEW) behaves bitwise like this branch's code
# at b9ea442 (OLD, the code every sweep run used): the unit fingerprint (models,
# objectives, sampler), end-to-end cases (consolidate_check.py: run ids, datasets,
# 200-step training with evaluations), re-scoring of a finished run; then the full
# test suite on NEW. Sized from GPU test job 60207586 (22 min, peak 1.25 GB).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
OLD=/n/scratch/users/e/edk005/nrfm/equiv-before NEW=/n/scratch/users/e/edk005/nrfm/main-consolidate
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
OUT=/n/scratch/users/e/edk005/nrfm/consolidate-check/$SLURM_JOB_ID && mkdir -p $OUT
RUN=$NRFM_OUT/ambient-flowmap/imf-s42-24d83765   # clean iMF, n = 100k, sigma = 1.0 set
nvidia-smi --query-gpu=name --format=csv,noheader
for T in OLD NEW; do
  cd ${!T} && echo "$T: $(git log -1 --oneline) $(git status --porcelain src configs scripts tests | wc -l) uncommitted"
  PYTHONPATH=${!T} python $D/fingerprint.py $OUT/unit-$T.json > /dev/null
  PYTHONPATH=${!T} python $D/consolidate_check.py ${T,,} $OUT/cases-$T.json $OUT/tmp
  PYTHONPATH=${!T} python -m scripts.eval $RUN --steps 1 2 4 > $OUT/eval-$T.json
done
cmp $OUT/unit-OLD.json $OUT/unit-NEW.json && echo "UNIT FINGERPRINTS IDENTICAL" || echo "UNIT FINGERPRINTS DIFFER"
cmp $OUT/cases-OLD.json $OUT/cases-NEW.json && echo "END-TO-END CASES IDENTICAL" || echo "END-TO-END CASES DIFFER"
cmp $OUT/eval-OLD.json $OUT/eval-NEW.json && echo "RE-SCORE IDENTICAL" || echo "RE-SCORE DIFFER"
cd $NEW && python -m pytest -v -rs
