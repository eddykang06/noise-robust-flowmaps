#!/usr/bin/env bash
# Checks that main's tree (NEW) behaves bitwise like OLD (default: this branch's code at
# b9ea442, the code every sweep run used; OLD, OLD_MODE from the environment otherwise): the
# unit fingerprint (models, objectives, sampler) and end-to-end cases (consolidate_check.py: run
# ids, datasets, 50-step training with an evaluation; differing fields listed); then the GPU
# tests selected by TESTS (pytest -k) on NEW, none if unset. Bare bones: the CPU parts are
# checked first by consolidate-check-cpu.sbatch. Short so it queues quickly; sized from GPU check
# 60636197 (12 min with the full test suite, under 2 GB).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
OLD=${OLD:-/n/scratch/users/e/edk005/nrfm/equiv-before} NEW=${NEW:-/n/scratch/users/e/edk005/nrfm/main-consolidate}
OLD_MODE=${OLD_MODE:-old}  # config layout of OLD: old (research/ configs) or new (configs/imf.yaml)
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
OUT=/n/scratch/users/e/edk005/nrfm/consolidate-check/$SLURM_JOB_ID && mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
for T in OLD NEW; do
  cd ${!T} && echo "$T: $(git log -1 --oneline) $(git status --porcelain src configs scripts tests | wc -l) uncommitted"
  PYTHONPATH=${!T} python $D/fingerprint.py $OUT/unit-$T.json > /dev/null
  M=new; [ $T = OLD ] && M=$OLD_MODE
  PYTHONPATH=${!T} python $D/consolidate_check.py $M $OUT/cases-$T.json $OUT/tmp/$T
done
cmp $OUT/unit-OLD.json $OUT/unit-NEW.json && echo "UNIT FINGERPRINTS IDENTICAL" || echo "UNIT FINGERPRINTS DIFFER"
cmp $OUT/cases-OLD.json $OUT/cases-NEW.json && echo "END-TO-END CASES IDENTICAL" || echo "END-TO-END CASES DIFFER"
python $D/diff_cases.py $OUT/cases-OLD.json $OUT/cases-NEW.json
if [ -n "${TESTS:-}" ]; then cd $NEW && python -m pytest -rs -m gpu -k "$TESTS"; fi
