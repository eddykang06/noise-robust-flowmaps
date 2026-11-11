#!/usr/bin/env bash
# Before the sweep: the data and ambient tests, then every arm for 300 steps in
# parallel, as sweep.sbatch runs them, into a scratch runs file.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
source research/experiments/ambient-flowmap/arms.sh
LOGS=$(mktemp -d /n/scratch/users/e/edk005/nrfm/smoke-XXXX)

run_arm() {
  arm_args $1
  python -m scripts.train "${ARGS[@]}" slug=smoke "out_root=$LOGS" "runs_dir=$LOGS" \
    train.steps=300 train.eval_every=300
}

python -m pytest -q tests/test_data.py tests/test_ambient_imf.py
all_arms run_arm || { tail -n 20 "$LOGS"/arm*.log; exit 1; }
test "$(wc -l < "$LOGS/smoke.jsonl")" -eq 5                       # one record per arm
python -c 'import json, sys; [json.loads(l) for l in sys.stdin]' < "$LOGS/smoke.jsonl"  # all whole
grep -h wasserstein2 "$LOGS"/arm*.log
ls "$LOGS"/smoke/*/true-vs-generated.png
