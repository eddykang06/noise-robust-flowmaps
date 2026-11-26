#!/usr/bin/env bash
# Downloads ImageNet (ILSVRC-2012) train and validation Parquet shards from Hugging Face
# (ILSVRC/imagenet-1k, gated; uses the stored `hf auth login` token) into the data root, for
# main's `scripts/prepare_data.py imagenet`. 146.5 + 6.7 GB; rerunning resumes. Memory is an
# estimate (no comparable job measured); CPU only.
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
DEST=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data/imagenet/hf
mkdir -p $DEST
hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/train-*" --include "data/validation-*" --include classes.py \
  --local-dir $DEST --max-workers 4
du -sh $DEST/data; ls $DEST/data | sed 's/-[0-9]*-of-.*//' | sort | uniq -c
