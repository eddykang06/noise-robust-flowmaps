#!/usr/bin/env bash
# Verifies the ImageNet posterior cache against our own SD-VAE ft-MSE encoding of the raw images
# (verify_latent_cache.py), then assembles main's memory-mapped val and train files from it
# (`prepare_data imagenet-assemble`), timed. CPU only. Memory: the CPU smoke's VAE encoding
# (60641517, 1.80 GB) plus a 3,572-row shard of posteriors (0.12 GB) and Arrow buffers.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NRFM_DATA=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data
D=$SLURM_SUBMIT_DIR/research/experiments/ambient-flowmap
cd /n/scratch/users/e/edk005/nrfm/main-parquet && git log -1 --oneline && git status --short
PYTHONPATH=. python $D/verify_latent_cache.py
for split in val train; do
  start=$(date +%s)
  python -m scripts.prepare_data imagenet-assemble --split $split
  echo "assembled $split in $(( $(date +%s) - start )) s"
done
python -c "
from src.datasets import imagenet
for split in ('val', 'train'):
    b = imagenet.load(split)
    print(split, tuple(b.x.shape), 'labels', int(b.cond.min()), int(b.cond.max()), 'classes', len(b.cond.unique()),
          'x mean %.3f std %.3f' % (b.x[:20000].mean(), b.x[:20000].std()), 'x_std median %.4f' % b.x_std[:2000].median())
"
