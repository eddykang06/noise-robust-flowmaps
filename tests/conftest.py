import os
import tempfile

import pytest
import torch

# configs/toy.yaml reads $NRFM_OUT; tests that train set out_root explicitly.
os.environ.setdefault("NRFM_OUT", tempfile.gettempdir())


def pytest_collection_modifyitems(items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs CUDA")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def gen():
    return lambda seed: torch.Generator().manual_seed(seed)
