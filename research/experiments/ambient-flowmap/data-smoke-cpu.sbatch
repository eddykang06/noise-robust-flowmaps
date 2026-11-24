#!/usr/bin/env bash
# CPU smoke test of main's real-data pipeline (tree NEW): prepares MNIST (download, metric
# classifier) and the FID Inception network into the lasting data root, then runs
# data_smoke_cpu.py (known-answer image metrics on MNIST; a fake ImageNet through the SD-VAE and
# Inception, in a temporary data root that shares the weights), then the CPU tests.
# Memory: 60641323 was killed at 5 GB encoding 64 images per VAE pass on CPU; now 4 per pass.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data NRFM_OUT=/n/scratch/users/e/edk005/nrfm
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
NEW=/n/scratch/users/e/edk005/nrfm/main-consolidate
OUT=/n/scratch/users/e/edk005/nrfm/data-smoke/$SLURM_JOB_ID && mkdir -p $OUT/root $NRFM_DATA/weights
cd $NEW && git log -1 --oneline && git status --porcelain | wc -l
python -m scripts.prepare_data mnist
python -m scripts.prepare_data inception
ln -s $NRFM_DATA/weights $OUT/root/weights && cp -r $NRFM_DATA/mnist $OUT/root/
NRFM_DATA=$OUT/root PYTHONPATH=$NEW python $D/data_smoke_cpu.py $OUT
python -m pytest -q -rs -m "not gpu"
