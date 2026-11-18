#!/usr/bin/env bash
# Resubmission of the clean-iMF dimension sweep 59981814 (dim-sweep.sbatch), which starts
# only if that job ends without success (e.g. its 2 h limit) and is dropped otherwise.
# Each run resumes from its last checkpoint; it does nothing if the tests had failed.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --dependency=afternotok:59981814
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
FIRST=$NRFM_OUT/ambient-flowmap/logs/59981814
LOGS=$NRFM_OUT/ambient-flowmap/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"
echo "$(git rev-parse --short HEAD); resuming 59981814; logs in $LOGS"

tail -1 "$FIRST/pytest.log" | grep -q " passed" && ! tail -1 "$FIRST/pytest.log" | grep -q "failed" \
  || { echo "59981814 did not pass its tests; not resuming"; exit 1; }

pids=()
for D in 2 8 16 512; do
  python -m scripts.train --config research/experiments/ambient-flowmap/spiral-imf.yaml \
    data.dim=$D model.dim=$D > "$LOGS/D$D.log" 2>&1 & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
grep -H "wasserstein2" "$LOGS"/D*.log | grep "step 50000"
exit $status
