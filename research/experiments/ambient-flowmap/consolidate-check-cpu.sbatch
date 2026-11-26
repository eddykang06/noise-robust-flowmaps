#!/usr/bin/env bash
# The CPU part of consolidate-check.sbatch (OLD, OLD_MODE from the environment), for a quick answer while GPUs are scarce:
# unit fingerprint (models, objectives, sampler), run ids, training sets and references
# of every case in OLD and NEW, then the CPU tests on NEW. Sized from CPU test job
# 60211648 (1 min, peak 0.70 GB).
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=1500M
#SBATCH --time=00:20:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
OLD=${OLD:-/n/scratch/users/e/edk005/nrfm/equiv-before} NEW=/n/scratch/users/e/edk005/nrfm/main-consolidate
OLD_MODE=${OLD_MODE:-old}  # config layout of OLD: old (research/ configs) or new (configs/imf.yaml)
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
OUT=/n/scratch/users/e/edk005/nrfm/consolidate-check/$SLURM_JOB_ID && mkdir -p $OUT
for T in OLD NEW; do
  cd ${!T} && echo "$T: $(git log -1 --oneline)"
  PYTHONPATH=${!T} python $D/fingerprint.py $OUT/unit-$T.json > /dev/null
  M=new; [ $T = OLD ] && M=$OLD_MODE
  PYTHONPATH=${!T} python $D/consolidate_check.py $M $OUT/cases-$T.json $OUT/tmp > /dev/null
done
cmp $OUT/unit-OLD.json $OUT/unit-NEW.json && echo "UNIT FINGERPRINTS IDENTICAL" || echo "UNIT FINGERPRINTS DIFFER"
cmp $OUT/cases-OLD.json $OUT/cases-NEW.json && echo "CASES IDENTICAL" || echo "CASES DIFFER"
python $D/diff_cases.py $OUT/cases-OLD.json $OUT/cases-NEW.json
grep -o '"run_id_eval2k": "[^"]*"' $OUT/cases-NEW.json
cd $NEW && python -m pytest -q -rs -m "not gpu"
