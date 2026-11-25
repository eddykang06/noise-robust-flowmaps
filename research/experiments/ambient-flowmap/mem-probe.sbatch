#!/usr/bin/env bash
# Peak GPU memory of one training step for main's MNIST and ImageNet configs (mem_probe.py), to
# choose a batch size that fits after MNIST batch 256 ran out of memory (60639945, 60704833).
# Host memory: 1.5x the CPU smoke peak (60641517, 1.80 GB).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export NRFM_OUT=/n/scratch/users/e/edk005/nrfm
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
cd /n/scratch/users/e/edk005/nrfm/main-consolidate && git log -1 --oneline
PYTHONPATH=. python $D/mem_probe.py
