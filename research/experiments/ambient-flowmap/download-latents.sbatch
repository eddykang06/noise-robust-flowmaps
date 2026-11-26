#!/usr/bin/env bash
# Downloads the SD-VAE ft-MSE posterior cache of ImageNet 256 (ChocolateDave/imagenet-256-sd-vae-latents:
# mean and std per image, iMF's preparation; 39.4 + 1.5 GB) into <root>/imagenet/posterior, where
# main's `prepare_data imagenet-download` puts it. Sized from download 60844562 (153 GB in 7 min).
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/nrfm/bin:$PATH"
DEST=/n/data1/hms/dbmi/zitnik/lab/users/edk005/nrfm-data/imagenet/posterior
mkdir -p $DEST
hf download ChocolateDave/imagenet-256-sd-vae-latents --repo-type dataset --include "data/*" --include "metadata/*" \
  --local-dir $DEST --max-workers 4
du -sh $DEST/data; ls $DEST/data | sed 's/-[0-9]*-of-.*//' | sort | uniq -c
