#!/usr/bin/env bash
# Scores final checkpoints with fresh sampling noise and fresh reference points: the run's seed
# is replaced by each of SEEDS for sampling only (training data untouched), so a value that
# depends on the one draw seed 42 gives shows up as spread across seeds. No training. Sized from
# re-score 60224909 (16 runs x 5 step counts: 18 min, peak 0.76 GB).
#   DESC=... research/bin/submit ambient-flowmap <this> --export=ALL,RUNS="<run> ...",SEEDS="1 2 3"
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1200M
#SBATCH --time=00:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
: "${RUNS:?set RUNS=\"<run name> ...\"}" "${SEEDS:?set SEEDS=\"<seed> ...\"}"
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
OUT=/n/scratch/users/e/edk005/nrfm/ambient-flowmap
LOGS=$OUT/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"
echo "$(git rev-parse --short HEAD); logs in $LOGS"

python - "$OUT" "$LOGS/resample.jsonl" "$SEEDS" $RUNS <<'EOF'
import json, sys
from dataclasses import replace
from pathlib import Path
from src.embed import frame
from src.training import load_run, reference, samples, score

out, path, seeds, runs = Path(sys.argv[1]), sys.argv[2], [int(s) for s in sys.argv[3].split()], sys.argv[4:]
with open(path, "w") as f:
    for run in runs:
        cfg, net = load_run(out / run)
        for seed in seeds:
            c = replace(cfg, seed=seed, sampler={**cfg.sampler, "steps": [1, 2, 8]})
            ref = reference(c).to("cuda")
            gen = samples(net, c, ref)
            row = {"run": run, "sample_seed": seed, **score(c, gen, ref),
                   **{f"far@{k}": (frame(x)[:, 2:].norm(dim=1) > 2).float().mean().item() for k, x in gen.items()}}
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
EOF
