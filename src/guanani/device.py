import warnings

import torch


def pick_device(requested: str | None = None) -> str:
    cuda = torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        warnings.warn("CUDA no disponible; usando CPU")
        return "cpu"
    return requested or ("cuda" if cuda else "cpu")
