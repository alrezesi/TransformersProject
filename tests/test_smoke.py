"""Smoke test: the environment works and torch is importable."""
import torch


def test_torch_works():
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)
