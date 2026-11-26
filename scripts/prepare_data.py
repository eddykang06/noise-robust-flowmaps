"""Download and preprocess a dataset, or the metric networks, into $NRFM_DATA (default data/).

    python -m scripts.prepare_data mnist                       # digits + the metric classifier (GPU)
    python -m scripts.prepare_data inception                   # FID Inception network
    python -m scripts.prepare_data imagenet-download            # SD-VAE posteriors, 41 GB
    python -m scripts.prepare_data imagenet-assemble --split train   # (and --split val)

ImageNet comes as precomputed SD-VAE ft-MSE posteriors (iMF's preparation, src/datasets/imagenet.py);
ImageNet's terms of use apply.
"""

import argparse
import urllib.request

from src.datasets import imagenet, mnist, root

INCEPTION = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"  # TorchScript

parser = argparse.ArgumentParser()
parser.add_argument("name", choices=["mnist", "inception", "imagenet-download", "imagenet-assemble"])
parser.add_argument("--split", choices=["train", "val"])
args = parser.parse_args()

if args.name == "mnist":
    mnist.download()
    print(f"metric classifier test accuracy {mnist.train_classifier():.4f}")
elif args.name == "inception":
    (root() / "weights").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(INCEPTION, root() / "weights" / "inception-2015-12-05.pt")
elif args.name == "imagenet-download":
    imagenet.download()
else:
    imagenet.assemble(args.split)
