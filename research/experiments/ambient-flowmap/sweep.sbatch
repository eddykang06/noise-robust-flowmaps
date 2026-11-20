#!/usr/bin/env bash
# First experiment (brief, eddy 90d5ebd): the five arms in parallel on one GPU, seed 42 only
# (human, 2026-11-18: no repeat runs at other seeds), the spiral observed in DIM dimensions
# (D chosen by the clean sweep, dim-sweep.sbatch). Arm 1 is that sweep's run at D, which
# resumes as already finished. 100k steps (configs). Sized from 60027106: four 100k runs in
# parallel took 1 h 54 min and peaked at 4.06 GB; the fixed-sigma sweep (60187230/34/87/88) took
# 1 h 35 min - 1 h 53 min and peaked at 4.4 GB.
#   DESC=... research/bin/submit ambient-flowmap <this> --export=ALL,DIM=<D>,SIGMA=<sigma>
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=02:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
source research/experiments/ambient-flowmap/arms.sh
LOGS=$NRFM_OUT/ambient-flowmap/logs/$SLURM_JOB_ID && mkdir -p "$LOGS"

: "${DIM:?set DIM}"
run_arm() {
  arm_args $1
  python -m scripts.train "${ARGS[@]}"
}

nvidia-smi --query-gpu=name --format=csv,noheader
echo "$(git rev-parse --short HEAD); D=$DIM; sigma=$SIGMA; logs in $LOGS"
all_arms run_arm
