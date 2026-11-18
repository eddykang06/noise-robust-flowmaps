#!/usr/bin/env bash
# Re-scores the four dimension-sweep checkpoints (job 59987798) at 1, 2, 3, 4 and 8
# sampling steps with scripts.eval (same reference and seeds as training), and saves
# 4096 samples per step count in the plane for the figure. No training.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
OUT=/n/scratch/users/e/edk005/nrfm/ambient-flowmap
LOGS=$OUT/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"
echo "$(git rev-parse --short HEAD); logs in $LOGS"

for run in 61bb6b98 d84ff5b3 16b83d21 c1d083a4; do
  echo "imf-s42-$run $(python -m scripts.eval $OUT/imf-s42-$run --steps 1 2 3 4 8)" | tee -a "$LOGS/eval.txt"
done

python - "$OUT" "$LOGS/plane-samples.json" <<'EOF'
import json, sys
from dataclasses import replace
from pathlib import Path
import torch
from src.embed import frame
from src.training import load_run, reference, samples

out, runs = {}, {"61bb6b98": 2, "d84ff5b3": 8, "16b83d21": 16, "c1d083a4": 512}
for run, dim in runs.items():
    cfg, net = load_run(Path(sys.argv[1]) / f"imf-s42-{run}")
    cfg = replace(cfg, sampler={**cfg.sampler, "steps": [1, 2, 3, 4, 8]})
    ref = reference(cfg).to("cuda")
    with torch.no_grad():
        gen = samples(net, cfg, ref)
    out[dim] = {s: frame(x)[:, :2].cpu().round(decimals=3).tolist() for s, x in gen.items()}
    out[dim]["true"] = frame(ref.x)[:, :2].cpu().round(decimals=3).tolist()
json.dump(out, open(sys.argv[2], "w"))
EOF
