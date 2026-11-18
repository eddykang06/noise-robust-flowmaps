#!/usr/bin/env bash
# Checks the embedded-toy PR (branch embed-toys) against main: the full test suite on both
# checkouts, then the same short training run on each, whose metric records must match.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=1
ROOT=/n/scratch/users/e/edk005/nrfm
OUT=$ROOT/pr-embed-check/$SLURM_JOB_ID && mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader

status=0
for tree in main-ref pr-embed; do
  cd "$ROOT/$tree" && echo "== $tree $(git rev-parse --short HEAD)"
  NRFM_OUT=$OUT/$tree python -m pytest -q -rs > "$OUT/$tree-pytest.log" 2>&1 || status=1
  tail -2 "$OUT/$tree-pytest.log"
  python -m scripts.train --config configs/toy.yaml slug=check out_root=$OUT/$tree runs_dir=$OUT/$tree \
    "corruption={name: gaussian, sigma: 0.3, clean_frac: 0.1}" train.steps=2000 train.eval_every=1000 \
    > "$OUT/$tree-train.log" 2>&1 || status=1
done
python - "$OUT" <<'EOF' || status=1
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
recs = {t: [{k: r[k] for k in ("run_id", "step", "metrics")}
            for r in map(json.loads, (out / t / "check.jsonl").read_text().splitlines())]
        for t in ("main-ref", "pr-embed")}
print("main:", recs["main-ref"]); print("pr:  ", recs["pr-embed"])
assert recs["main-ref"] == recs["pr-embed"], "training records differ"
print("training records identical")
EOF
exit $status
