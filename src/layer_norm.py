"""Layer normalization written from scratch."""
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # learnable shift

    def forward(self, x: Tensor) -> Tensor:
        """x: [..., D] -> [..., D]; each token is normalized on its own."""
        mean = x.mean(dim=-1, keepdim=True)
        # Biased variance (divide by D), same as torch.nn.LayerNorm.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta