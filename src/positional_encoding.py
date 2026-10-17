"""Sinusoidal positional encoding, stored as a buffer."""
import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding")
        self.dropout = nn.Dropout(dropout)

        # Positions 0..max_len-1 as a column vector: [max_len, 1]
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # Frequencies for each sin/cos pair, computed in log-space for stability:
        # 1 / 10000^(2i/D) == exp(-(2i/D) * ln(10000)).  Shape: [D/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)   # even dims
        pe[:, 1::2] = torch.cos(positions * div_term)   # odd dims

        # Buffer: moves with the module (.to(device), state_dict) but has no gradient.
        # Shape [1, max_len, D] so it broadcasts over the batch dimension.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, D] -> [B, T, D] (x plus the first T encodings)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)