#!/usr/bin/env bash
# Encodes ImageNet validation shard 0 with main's prepare_data (timed, to size the full encoding)
# into the real data root, then checks it (imagenet_check.py): labels match the Parquet source,
# SD-VAE round-trip PSNR, a labelled grid of decoded images. Memory: 1.5x GPU smoke 60707401
# (1.54 GB, VAE included).
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
cd /n/scratch/users/e/edk005/nrfm/main-parquet && git log -1 --oneline && nvidia-smi --query-gpu=name --format=csv,noheader
start=$(date +%s)
python -m scripts.prepare_data imagenet-encode --split val --shards 0:1
echo "encoded val shard 0 in $(( $(date +%s) - start )) s"
PYTHONPATH=. python $D/imagenet_check.py val 0 /n/scratch/users/e/edk005/nrfm/imagenet-check-$SLURM_JOB_ID.png
