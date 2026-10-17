"""Token embedding: nn.Embedding scaled by sqrt(d_model)."""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        # Attribute name `embedding` is used by the tests.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Default init is N(0, 1); after the sqrt(D) scaling the output std would
        # be ~sqrt(D). Using std = D^-0.5 makes the scaled output have std ~ 1.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

        # The re-init above overwrote the zero padding row, so zero it again.
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [B, T] (int64) -> [B, T, D]"""
        return self.embedding(tokens) * math.sqrt(self.d_model)