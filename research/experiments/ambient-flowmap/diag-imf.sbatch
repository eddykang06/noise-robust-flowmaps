#!/usr/bin/env bash
# Why iMF does not learn the clean spiral (sweep 59276661, arm 1): re-score that run with
# more sampling steps, and train three variants of its lr / EMA, all on clean data.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
LOGS=$NRFM_OUT/ambient-flowmap/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"
echo "$(git rev-parse --short HEAD); logs in $LOGS"

# Does the network learn the flow, if not the one-step jump?
python -m scripts.eval $NRFM_OUT/ambient-flowmap/imf-s42-db4d1a3c --steps 1 2 8 50

train() {  # <name> <overrides...>
  python -m scripts.train --config research/experiments/ambient-flowmap/spiral-imf.yaml \
    "sampler.steps=[1, 2, 8, 50]" "${@:2}" > "$LOGS/$1.log" 2>&1
}
pids=()
train lr1e-3-ema9999 train.lr=1.0e-3 & pids+=($!)
train lr1e-3-ema999 train.lr=1.0e-3 train.ema=0.999 & pids+=($!)
train lr1e-4-ema999 train.ema=0.999 & pids+=($!)
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
grep -H wasserstein2 "$LOGS"/*.log
exit $status
