#!/usr/bin/env bash
# Does conditioning the toy MLP on t fix iMF on the clean spiral? (diag-imf.sbatch: without
# t, no lr / EMA learns it.) Official lr / EMA, and lr 1e-3 / EMA 0.999, 50k steps.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
LOGS=$NRFM_OUT/ambient-flowmap/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"
echo "$(git rev-parse --short HEAD); logs in $LOGS"
python -m pytest -q tests/test_imf_reference.py

train() {  # <name> <overrides...>
  python -m scripts.train --config research/experiments/ambient-flowmap/spiral-imf.yaml \
    model.condition_on_t=true "sampler.steps=[1, 2, 8, 50]" "${@:2}" > "$LOGS/$1.log" 2>&1
}
pids=()
train t-official & pids+=($!)
train t-lr1e-3-ema999 train.lr=1.0e-3 train.ema=0.999 & pids+=($!)
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
grep -H wasserstein2 "$LOGS"/*.log
exit $status
