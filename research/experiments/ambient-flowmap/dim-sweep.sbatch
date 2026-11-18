#!/usr/bin/env bash
# Pixel MeanFlow toy setup (arXiv 2601.22158, Sec. 5) for exact iMF: the clean two-armed
# spiral observed in D dimensions (src/embed.py), D in {2, 8, 16, 512}, seed 42, 50k steps.
# The test suite ran in the first submission, 59981814 (64 passed; the one failure is
# test_learns_the_checkerboard_in_one_step, exact iMF on a 2-D toy, failing since
# 59185847 on main), so this job only trains.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=02:00:00
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
for D in 2 8 16 512; do
  python -m scripts.train --config research/experiments/ambient-flowmap/spiral-imf.yaml \
    data.dim=$D model.dim=$D > "$LOGS/D$D.log" 2>&1 & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
grep -H "wasserstein2" "$LOGS"/D*.log | grep "step 50000"
exit $status
