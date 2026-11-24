"""Download and preprocess a dataset, or the metric networks, into $NRFM_DATA (default data/).

    python -m scripts.prepare_data mnist                       # digits + the metric classifier (GPU)
    python -m scripts.prepare_data inception                   # FID Inception network
    python -m scripts.prepare_data imagenet --src <dir> --split train   # SD-VAE latents (GPU)

ImageNet is read from `<dir>/<class folder>/<image>`, e.g. the ILSVRC-2012 train or val
images sorted into one folder per WordNet ID.
"""

import argparse
import urllib.request
from pathlib import Path

from src.datasets import imagenet, mnist, root

INCEPTION = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"  # TorchScript

parser = argparse.ArgumentParser()
parser.add_argument("name", choices=["mnist", "inception", "imagenet"])
parser.add_argument("--src", type=Path)
parser.add_argument("--split", choices=["train", "val"])
args = parser.parse_args()

if args.name == "mnist":
    mnist.download()
    print(f"metric classifier test accuracy {mnist.train_classifier():.4f}")
elif args.name == "inception":
    (root() / "weights").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(INCEPTION, root() / "weights" / "inception-2015-12-05.pt")
else:
    imagenet.prepare(args.src, args.split)
