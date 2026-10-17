"""Position-wise feed-forward network."""
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "relu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # expand
        self.linear2 = nn.Linear(d_ff, d_model)    # project back
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"unknown activation: {activation!r}")

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, D] -> [B, T, D]. Applied to every position independently."""
        # [B, T, D] -> [B, T, d_ff] -> activation -> dropout -> [B, T, D]
        return self.linear2(self.dropout(self.activation(self.linear1(x))))