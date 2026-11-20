#!/usr/bin/env bash
# Retrains one arm of the fixed-sigma sweep with evaluation every EVAL_EVERY steps, to find when
# its one-step map breaks (ambient Gaussian, sigma = 1.0: 5.507 at 100k after 0.94-1.29 up to
# 90k). Evaluation does not touch the training RNG, so only the eval cadence differs from the
# sweep run; the run gets its own id. Sized from 60207443 (four arms in parallel: 1 h 47 min,
# peak 4.2 GB, so about 1 GB per arm).
#   DESC=... research/bin/submit ambient-flowmap <this> --export=ALL,DIM=64,SIGMA=1.0,ARM=3,EVAL_EVERY=2000
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=02:30:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
: "${DIM:?set DIM}" "${ARM:?set ARM}" "${EVAL_EVERY:?set EVAL_EVERY}"
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
source research/experiments/ambient-flowmap/arms.sh
nvidia-smi --query-gpu=name --format=csv,noheader
echo "$(git rev-parse --short HEAD); D=$DIM; sigma=$SIGMA; arm=$ARM; eval every $EVAL_EVERY"
arm_args $ARM
python -m scripts.train "${ARGS[@]}" train.eval_every=$EVAL_EVERY
