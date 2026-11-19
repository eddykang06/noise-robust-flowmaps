#!/usr/bin/env bash
# Pixel MeanFlow toy setup (arXiv 2601.22158, Sec. 5) for exact iMF: the clean two-armed
# spiral observed in each D of DIMS (src/embed.py), in parallel, seed 42, 100k steps; then
# re-scores the checkpoints at 1-8 sampling steps (more-steps.sbatch).
#   DESC=... research/bin/submit ambient-flowmap <this> --export=ALL,DIMS="2 8 16 32"
# Header sized for four D: 60027106 (D = 2, 8, 16, 32) took 1 h 54 min, peak 4.06 GB.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
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
nvidia-smi --query-gpu=name --format=csv,noheader

pids=()
for D in ${DIMS:?set DIMS}; do
  python -m scripts.train --config research/experiments/ambient-flowmap/spiral-imf.yaml \
    data.dim=$D model.dim=$D > "$LOGS/D$D.log" 2>&1 & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
grep -H "wasserstein2" "$LOGS"/D*.log | grep "step 100000"
[ $status = 0 ] || exit $status
RUNS=$(for D in $DIMS; do echo "$(grep -om1 'imf-s42-[0-9a-f]*' "$LOGS/D$D.log"):$D"; done)
RUNS="$RUNS" bash research/experiments/ambient-flowmap/more-steps.sbatch
