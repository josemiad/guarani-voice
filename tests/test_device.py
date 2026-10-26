import pytest
import torch

from guanani.device import pick_device


def test_cuda_requested_but_missing_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA"):
        assert pick_device("cuda") == "cpu"


def test_default_and_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"
    assert pick_device("cpu") == "cpu"
