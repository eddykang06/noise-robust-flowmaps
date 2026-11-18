#!/usr/bin/env bash
# Shared-pipeline cleanup on main (sample_clean -> src/toy.py; per-residue masks): full test
# suite on the new and the pre-merge main, then parity.py fingerprints of both, compared bitwise.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=1
ROOT=/n/scratch/users/e/edk005/nrfm
PARITY=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap/parity.py
OUT=$ROOT/parity/$SLURM_JOB_ID && mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader

status=0
for tree in main-ref main-work; do
  cd "$ROOT/$tree" && echo "== $tree $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) changed files"
  NRFM_OUT=$OUT/$tree python -m pytest -q -rs -p no:cacheprovider > "$OUT/$tree-pytest.log" 2>&1 || true
  grep -E "^(FAILED|ERROR)|passed|failed" "$OUT/$tree-pytest.log" | tail -5
  NRFM_OUT=$OUT/$tree python "$PARITY" dump "$OUT/$tree.pt" || status=1
done
python "$PARITY" compare "$OUT/main-ref.pt" "$OUT/main-work.pt" || status=1
exit $status
