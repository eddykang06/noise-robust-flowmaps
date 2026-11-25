#!/usr/bin/env bash
# Scores the raw (non-EMA) weights of GPU smoke 60707401's 2,000-step MNIST run (raw_net_check.py),
# whose EMA (0.9999) is still mostly the initial weights, to see whether MNIST training learns.
# Host memory: 1.5x memory probe 60706992's peak (1.2 GB), rounded up.
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
RUN=$NRFM_OUT/data-smoke/60707401/runs/data-smoke/imf-s42-a84ecfee
cd /n/scratch/users/e/edk005/nrfm/main-consolidate && git log -1 --oneline
PYTHONPATH=. python $D/raw_net_check.py $RUN $NRFM_OUT/data-smoke/60707401/raw-net.png
