"""Download and preprocess a dataset, or the metric networks, into $NRFM_DATA (default data/).

    python -m scripts.prepare_data mnist                       # digits + the metric classifier (GPU)
    python -m scripts.prepare_data inception                   # FID Inception network
    python -m scripts.prepare_data imagenet-download            # Parquet shards from Hugging Face
    python -m scripts.prepare_data imagenet-encode --split train --shards 0:40   # SD-VAE latents (GPU)
    python -m scripts.prepare_data imagenet-assemble --split train

ImageNet comes from Hugging Face's ILSVRC/imagenet-1k (accept its terms, then `hf auth login`).
`imagenet-encode` takes a range of shards (default all: train has 294, val 14) and skips shards
already encoded, so several short jobs can share the work; `imagenet-assemble` needs them all.
"""

import argparse
import urllib.request

from src.datasets import imagenet, mnist, root

INCEPTION = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"  # TorchScript

parser = argparse.ArgumentParser()
parser.add_argument("name", choices=["mnist", "inception", "imagenet-download", "imagenet-encode", "imagenet-assemble"])
parser.add_argument("--split", choices=["train", "val"])
parser.add_argument("--shards", default=":", help="first:last, a Python-style range of shards")
args = parser.parse_args()

if args.name == "mnist":
    mnist.download()
    print(f"metric classifier test accuracy {mnist.train_classifier():.4f}")
elif args.name == "inception":
    (root() / "weights").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(INCEPTION, root() / "weights" / "inception-2015-12-05.pt")
elif args.name == "imagenet-download":
    imagenet.download()
elif args.name == "imagenet-encode":
    first, last = (int(v) if v else None for v in args.shards.split(":"))
    imagenet.encode(args.split, first or 0, last)
else:
    imagenet.assemble(args.split)
