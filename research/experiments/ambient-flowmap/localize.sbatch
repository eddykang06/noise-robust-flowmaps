#!/usr/bin/env bash
# Retrains one arm of the fixed-sigma sweep with config OVERRIDES (e.g. evaluation every 2k steps), to find when
# its one-step map breaks (ambient Gaussian, sigma = 1.0: 5.507 at 100k after 0.94-1.29 up to
# 90k). Evaluation does not touch the training RNG, so only the overrides differ from the
# sweep run; the run gets its own id. Sized from 60233034 (one arm, eval every 2k: 2 h 00 min,
# peak 1.16 GB; 60246613: 1 h 58 min, 1.11 GB; 60256817: 1 h 45 min, 1.04 GB). Time allows a 4x batch.
#   DESC=... research/bin/submit ambient-flowmap <this> --export=ALL,DIM=64,SIGMA=1.0,ARM=3,OVERRIDES="train.eval_every=2000"
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1700M
#SBATCH --time=04:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
: "${DIM:?set DIM}" "${ARM:?set ARM}" "${OVERRIDES:?set OVERRIDES}"
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
source research/experiments/ambient-flowmap/arms.sh
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$(git rev-parse --short HEAD); D=$DIM; sigma=$SIGMA; arm=$ARM; overrides: $OVERRIDES"
arm_args $ARM
python -m scripts.train "${ARGS[@]}" $OVERRIDES
