"""Peak GPU memory of one fp32 training step (forward with JVP, backward, Adam) of main's MNIST and
ImageNet configs, for a few patch sizes and batch sizes. Run from main's tree with PYTHONPATH set."""
from functools import partial

import torch

from src.config import kwargs, load
from src.data import Batch
from src.models import MODELS
from src.objectives import OBJECTIVES

CASES = [("configs/mnist.yaml", p, b) for p in (2, 4) for b in (32, 64, 128, 256)] + \
        [("configs/imagenet.yaml", 2, b) for b in (16, 32, 64, 128)]
print(torch.cuda.get_device_name(), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
for path, patch, bs in CASES:
    cfg = load(path, ["slug=probe", f"model.patch_size={patch}"])
    m = cfg.model
    net = MODELS[m["name"]](**kwargs(m)).cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_fn = partial(OBJECTIVES[cfg.objective["name"]], **kwargs(cfg.objective))
    batch = Batch(x=torch.randn(bs, m["in_channels"], m["input_size"], m["input_size"]), sigma=torch.zeros(bs),
                  cond=torch.randint(m["num_classes"], (bs,))).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        loss, _ = loss_fn(net, batch)
        loss.backward()
        opt.step()
        peak = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    except torch.OutOfMemoryError:
        peak = "OOM"
    print(f"{path} patch {patch} batch {bs}: {peak}", flush=True)
    del net, opt, batch
    loss = None
    torch.cuda.empty_cache()
